// app/chats/[chatId].tsx
import React, { useState, useEffect } from "react";
import { 
  View, Text, TouchableOpacity, FlatList, Image, 
  TextInput, KeyboardAvoidingView, Platform 
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { useLocalSearchParams } from "expo-router";
import { GetUser, useGetMessages } from "@/lib/kubb";
import { DecryptMessageBody, useGetNormalKeysChatId, useGetClientChatId } from "@/lib/kubb-he";
import { useMessagesAPI } from "@/hooks/use-websockets-messages";
import dayjs from 'dayjs';


export default function ConversationScreen() {
  const params = useLocalSearchParams<{
    sender?: string;
    receiver?: string;
    chatId: string;
  }>();

  let sender: GetUser | undefined = undefined;
  let receiver: GetUser | undefined = undefined;
  const chatId = Number(params.chatId);

  try {
    if (params.sender) sender = JSON.parse(params.sender);
    if (params.receiver) receiver = JSON.parse(params.receiver);
  } catch (err) {
    console.warn("Failed to parse sender/receiver JSON", err);
  }

  const [text, setText] = useState("");
  const [image, setImage] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  
  const checkFilesAndRequest = (chatID : number) => {
        try {
          useGetNormalKeysChatId(chatID).refetch();
          useGetClientChatId(chatID).refetch();
        } catch (error) {
          console.error("Error sending request:", error);
        }
  };

  useEffect(() => {
    if (chatId) {
      checkFilesAndRequest(chatId);
    }
  }, [chatId]);

  const { 
    messages, 
    setMessages, 
    decryptData, 
    sendEncryptedMessage, 
    setupWebSocketConnection, 
    socketRef, 
    sendTypingEvent 
  } = useMessagesAPI({
    sender,
    targetUser: receiver?.id!,
    chatID: chatId,
    offset,
  });

  const getMessagesQuery = useGetMessages(chatId, {
    skip: offset,
    sort_by: "id",
    order_by: "DESC",
  });

  const fetchMessages = async () => {
    setLoading(true);
    try {
      const { data } = await getMessagesQuery.refetch({});
      const newMessages = data?.array || [];

      const decryptedMessages = await Promise.all(
        newMessages.map(async (m) => {
          const encryptedMessage: DecryptMessageBody = {
            id: m.id!,
            chat_id: m.chat_id!,
            sender_id: m.sender_id!,
            sender_name: m.sender_name!,
            receiver_id: m.receiver_id!,
            content: m.content!,
            iv: m.iv!,
            image: m.image,
            image_to_classify: (m as any).image_to_classify!,
            type: m.type!,
            is_typing: m.is_typing!,
            timestamp: m.timestamp!,
            classification_result: m.classification_result!,
          };
          const decrypted = await decryptData(encryptedMessage);
          return { ...m, ...decrypted };
        })
      );

      setMessages(prev => {
        const existingIds = new Set(prev.map(m => m.id));
        const filtered = decryptedMessages.filter(m => !existingIds.has(m.id));
        const combined = [...prev, ...filtered];

        // Sort ascending by timestamp
        return combined.sort((a, b) => {
          const timeA = a.timestamp ? new Date(a.timestamp).getTime() : 0;
          const timeB = b.timestamp ? new Date(b.timestamp).getTime() : 0;
          return timeA - timeB;
        });
      });

      setTotal(data?.meta?.total ?? 0);
    } catch (err) {
      console.error("Fetch messages failed", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setMessages([]);
    setOffset(0);
    fetchMessages(); 
  }, [chatId]);

  useEffect(() => {
    setupWebSocketConnection();
    fetchMessages();
    return () => {
      if (socketRef.current) socketRef.current.close();
    };
  }, [chatId]);

  useEffect(() => {
    if (offset === 0) return;
    fetchMessages();
  }, [offset]);

  const handleSubmitMessage = async () => {
    if (!text.trim() && !image) return;
    try {
      await sendEncryptedMessage(text.trim(), image ?? "");
      setText("");
      setImage(null);
    } catch (err) {
      console.error("Failed to send message:", err);
    }
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 0.7,
      base64: true, 
    });

    if (!result.canceled && result.assets.length > 0) {
      const base64Image = result.assets[0].base64; 
      setImage(`data:image/jpeg;base64,${base64Image}`);
    }
  };

  const renderMessage = ({ item }: any) => {
    const isOwn = item.sender_id === sender?.id;
    console.log(item.classification_result)
    const shouldBlur =
      item.classification_result === "Pending" ||
      item.classification_result === "true";

    let imageSrc: string | undefined;
    if (item.image) {
      if (item.image.startsWith("data:image")) {
        imageSrc = item.image; 
      } else {
        imageSrc = `data:image/png;base64,${item.image}`; // fallback default
      }
    }

    return (
        <View
          style={{
            flexDirection: 'row',
            justifyContent: isOwn ? 'flex-end' : 'flex-start',
            paddingHorizontal: 8,
            marginVertical: 2,
          }}
        >
          <View
            style={{
              maxWidth: '75%',
              padding: 10,
              borderRadius: 12,
              backgroundColor: isOwn ? '#DCF8C6' : '#E5E5EA',
            }}
          >
            {imageSrc && (
              <Image
                source={{ uri: imageSrc }}
                style={{
                  width: 180,
                  height: 120,
                  borderRadius: 8,
                  marginBottom: 4,
                  ...(shouldBlur ? { blurRadius: 5 } : {}),
                }}
              />
            )}

            {item.content && (
              <Text style={{ fontSize: 15, marginBottom: 4 }}>
                {item.content}
              </Text>
            )}

            <Text
              style={{
                fontSize: 10,
                color: '#666',
                textAlign: 'right',
              }}
            >
              {dayjs(item.timestamp).format('YYYY-MM-DD HH:mm:ss')}
            </Text>
          </View>
        </View>
      );
    };
  
  return (
    <KeyboardAvoidingView
      style={{ flex: 1, backgroundColor: "black" }}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
      keyboardVerticalOffset={Platform.OS === "ios" ? 90 : 0}
    >
      {/* Main layout */}
      <View style={{ flex: 1 }}>
        {/* Messages */}
        <FlatList
          style={{ flex: 1 }}
          contentContainerStyle={{
            flexGrow: 1,
            justifyContent: messages.length === 0 ? "flex-end" : "flex-start",
          }}
          data={messages}
          renderItem={renderMessage}
          keyExtractor={(item, idx) => String(item.id ?? idx)}
          onEndReached={() => {
              if (!loading && total !== null && offset < total) {
                setOffset(prev => Math.min(prev + 10, total));
              }
            }}
          onEndReachedThreshold={0.1}
        />

        {/* Input bar pinned to bottom */}
        <View
          style={{
            flexDirection: "row",
            alignItems: "center",
            paddingHorizontal: 10,
            paddingVertical: 20,
            borderTopWidth: 1,
            borderColor: "#ccc",
            backgroundColor: "#fff",
          }}
        >
          {/* Image Picker Button */}
          <TouchableOpacity onPress={pickImage} style={{ marginRight: 8 }}>
            <Text style={{ fontSize: 22 }}>📷</Text>
          </TouchableOpacity>

          {/* Text Input */}
          <TextInput
            value={text}
            onChangeText={(val) => {
              setText(val);
              sendTypingEvent(true);
            }}
            placeholder="Type a message"
            style={{
              flex: 1,
              paddingVertical: 8,
              paddingHorizontal: 12,
              backgroundColor: "#f1f1f1",
              borderRadius: 20,
              fontSize: 16,
            }}
          />

          {/* Send Button */}
          <TouchableOpacity
            onPress={handleSubmitMessage}
            style={{
              marginLeft: 8,
              backgroundColor: "#007AFF",
              borderRadius: 20,
              paddingHorizontal: 14,
              paddingVertical: 8,
            }}
          >
            <Text style={{ color: "white", fontWeight: "bold" }}>Send</Text>
          </TouchableOpacity>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}


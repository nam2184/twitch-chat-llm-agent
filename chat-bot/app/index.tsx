import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  FlatList,
  TextInput,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from "react-native";
import * as DocumentPicker from "expo-document-picker";
import { useChatEndpointApiChatPost, useUploadPdfApiUploadPdfPost } from "@/lib/kubb"; // generated client
import dayjs from "dayjs";

type ChatMessage = {
  role: "user" | "assistant" | "system";
  content: string;
};

export default function ConversationScreen() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const flatListRef = useRef<FlatList>(null);

  // kubb-generated hooks
  const chatMutation = useChatEndpointApiChatPost();
  const uploadPdfMutation = useUploadPdfApiUploadPdfPost();

  useEffect(() => {
    // Auto-scroll to bottom when a new message arrives
    flatListRef.current?.scrollToEnd({ animated: true });
  }, [messages]);

  const sendMessage = async () => {
    const content = text.trim();
    if (!content) return;

    const userMsg: ChatMessage = { role: "user", content };
    setMessages((prev) => [...prev, userMsg]);
    setText("");
    setLoading(true);

    try {
      const body = { messages: [...messages, userMsg] };
      const response = await chatMutation.mutateAsync({ data: body });

      const botMsg: ChatMessage = {
        role: "assistant",
        content: response.response,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      console.error("Chat error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadPdf = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: "application/pdf",
      });
      if (result.canceled || !result.assets.length) return;

      const file = result.assets[0];
      const formData = new FormData();
      formData.append("file", {
        uri: file.uri,
        type: "application/pdf",
        name: file.name,
      } as any);

      const uploadRes = await uploadPdfMutation.mutateAsync({ data: formData });
      const systemMsg: ChatMessage = {
        role: "system",
        content: uploadRes.message || "PDF uploaded successfully.",
      };
      setMessages((prev) => [...prev, systemMsg]);
    } catch (err) {
      console.error("PDF upload failed:", err);
    }
  };

  const renderMessage = ({ item }: { item: ChatMessage }) => {
    const isUser = item.role === "user";
    const bgColor = isUser ? "#DCF8C6" : item.role === "assistant" ? "#E5E5EA" : "#F1F0F0";

    return (
      <View
        style={{
          flexDirection: "row",
          justifyContent: isUser ? "flex-end" : "flex-start",
          paddingHorizontal: 10,
          marginVertical: 4,
        }}
      >
        <View
          style={{
            maxWidth: "80%",
            backgroundColor: bgColor,
            padding: 10,
            borderRadius: 10,
          }}
        >
          <Text style={{ fontSize: 15 }}>{item.content}</Text>
          <Text
            style={{
              fontSize: 10,
              color: "#666",
              marginTop: 4,
              textAlign: isUser ? "right" : "left",
            }}
          >
            {dayjs().format("HH:mm")}
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
      <View style={{ flex: 1 }}>
        {/* Messages */}
        <FlatList
          ref={flatListRef}
          data={messages}
          renderItem={renderMessage}
          keyExtractor={(_, idx) => idx.toString()}
          contentContainerStyle={{
            flexGrow: 1,
            justifyContent: messages.length === 0 ? "flex-end" : "flex-start",
            backgroundColor: "white",
          }}
        />

        {/* Input Bar */}
        <View
          style={{
            flexDirection: "row",
            alignItems: "center",
            borderTopWidth: 1,
            borderColor: "#ccc",
            backgroundColor: "#fff",
            paddingHorizontal: 10,
            paddingVertical: 8,
          }}
        >
          <TouchableOpacity onPress={handleUploadPdf} style={{ marginRight: 10 }}>
            <Text style={{ fontSize: 22 }}>📄</Text>
          </TouchableOpacity>

          <TextInput
            value={text}
            onChangeText={setText}
            placeholder="Type a message..."
            style={{
              flex: 1,
              paddingVertical: 8,
              paddingHorizontal: 12,
              backgroundColor: "#f1f1f1",
              borderRadius: 20,
              fontSize: 16,
            }}
          />

          <TouchableOpacity
            onPress={sendMessage}
            style={{
              marginLeft: 8,
              backgroundColor: "#007AFF",
              borderRadius: 20,
              paddingHorizontal: 14,
              paddingVertical: 8,
            }}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <Text style={{ color: "white", fontWeight: "bold" }}>Send</Text>
            )}
          </TouchableOpacity>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}

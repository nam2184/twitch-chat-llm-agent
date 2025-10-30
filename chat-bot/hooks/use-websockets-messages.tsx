import React, { useState, useRef } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { GetUser, Message } from "@/lib/kubb";
import { DecryptMessageBody, EncryptedMessage, usePostDecrypt, usePostEncrypt } from "@/lib/kubb-he";

interface MessageProps {
  sender?: GetUser | undefined;
  targetUser: number;
  chatID: number;
  offset: number;
}

const useMessagesAPI = ({ targetUser, chatID, offset, sender }: MessageProps) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const socketRef = useRef<WebSocket | null>(null);
  const [socketStatus, setSocketStatus] = useState<boolean>(false);
  const [isTypingMessage, setIsTyping] = useState<boolean | undefined>(false);
  const postEncryptMutation = usePostEncrypt();
  const postDecryptMutation = usePostDecrypt();

  const setupWebSocketConnection = async () => {
    try {
      const accessToken = await AsyncStorage.getItem("access_token");
      if (!accessToken) {
        console.warn("No access token found for WebSocket");
        return;
      }

      const socket = new WebSocket(
        `wss://khanhmychattypi.win/api/v1/ws/${chatID}?access_token=${accessToken}`
      );

      socket.onopen = () => {
        console.log("WebSocket connected!");
        setSocketStatus(true);
      };

      socket.onclose = () => {
        console.log("WebSocket disconnected!");
        setSocketStatus(false);
        setTimeout(setupWebSocketConnection, 3000); // try reconnect
      };

      socket.onmessage = (event) => {
        const incoming: Message = JSON.parse(event.data);

        (async () => {
          if (incoming?.type === "typing") {
            setIsTyping(incoming.is_typing);
            return;
          }

          const encryptedMessage: DecryptMessageBody = {
            id: incoming.id!,
            chat_id: incoming.chat_id!,
            sender_id: incoming.sender_id!,
            sender_name: incoming.sender_name!,
            receiver_id: incoming.receiver_id!,
            content: incoming.content!,
            iv: incoming.iv!,
            image: incoming.image,
            image_to_classify: (incoming as any).image_to_classify!,
            type: incoming.type!,
            is_typing: incoming.is_typing!,
            timestamp: incoming.timestamp!,
            classification_result: incoming.classification_result!,
          };

          const decrypted = await decryptData(encryptedMessage);

          const newMessage: Message = {
            ...incoming,
            image: decrypted?.image,
            content: decrypted?.content,
            classification_result: decrypted?.classification_result,
          };

          setMessages((prev) => {
            const exists = prev.some((msg) => msg.id === newMessage.id);
            if (exists) {
              return prev.map((msg) =>
                msg.id === newMessage.id ? { ...msg, ...newMessage } : msg
              );
            } else {
              return [...prev, newMessage];
            }
          });
        })();
      };

      socket.onerror = (error) => {
        console.error("WebSocket error:", error);
      };

      socketRef.current = socket;
    } catch (err) {
      console.error("Error setting up WebSocket:", err);
    }
  };

  const sendMessage = async (content: string, image: string) => {
    if ((!content.trim() && image === "") || !socketRef.current) return;

    const message: Message = {
      chat_id: chatID,
      type: "message",
      sender_id: sender?.id,
      sender_name: sender?.username,
      receiver_id: targetUser,
      content,
      image,
      timestamp: new Date().toISOString(),
      is_typing: false,
    };

    try {
      socketRef.current.send(JSON.stringify(message));
      console.log("Sending message:", JSON.stringify(message));
    } catch (error) {
      console.error("Error sending message through WebSocket:", error);
    }
  };

  const sendEncryptedMessage = async (content: string, image: string) => {
    if ((!content.trim() && image === "") || !socketRef.current) return;

    const message: EncryptedMessage = {
      chat_id: chatID,
      type: "message",
      sender_id: sender?.id!,
      sender_name: sender?.username!,
      receiver_id: targetUser,
      content,
      image,
      timestamp: new Date().toISOString(),
      is_typing: false,
      classification_result: "",
    };

    const encryptedMessage = await encryptData(message);
    if (!encryptedMessage) return;

    try {
      socketRef.current.send(JSON.stringify(encryptedMessage));
      console.log("Sending encrypted message:", JSON.stringify(encryptedMessage));
    } catch (error) {
      console.error("Error sending encrypted message:", error);
    }
  };

  const sendTypingEvent = (is: boolean) => {
    if (!socketRef.current) return;

    const typingEvent: Message = {
      chat_id: chatID,
      sender_id: sender?.id,
      type: "typing",
      sender_name: sender?.username,
      receiver_id: targetUser,
      timestamp: new Date().toISOString(),
      is_typing: is,
    };

    try {
      socketRef.current.send(JSON.stringify(typingEvent));
      console.log("Typing event sent:", typingEvent);
    } catch (error) {
      console.error("Error sending typing event:", error);
    }
  };

  const encryptData = async (data: EncryptedMessage) => {
    try {
      const response = await postEncryptMutation.mutateAsync({
        data: {
          content: data.content!,
          image: data.image,
          chat_id: data.chat_id!,
          sender_id: data.sender_id!,
          type: data.type!,
          sender_name: data.sender_name!,
          receiver_id: data.receiver_id!,
          is_typing: data.is_typing,
          timestamp: data.timestamp!,
        },
      });
      return response;
    } catch (error) {
      console.error("Error encrypting data:", error);
      return null;
    }
  };

  const decryptData = (data: DecryptMessageBody) => {
    try {
      return postDecryptMutation.mutateAsync({
        data: {
          id: data.id,
          content: data.content!,
          image: data.image,
          image_to_classify: data.image_to_classify,
          chat_id: data.chat_id!,
          sender_id: data.sender_id!,
          iv: data.iv!,
          type: data.type!,
          sender_name: data.sender_name!,
          receiver_id: data.receiver_id!,
          is_typing: data.is_typing,
          timestamp: data.timestamp!,
          classification_result: data.classification_result!,
        },
      });
    } catch (error) {
      console.error("Error decrypting data:", error);
      return null;
    }
  };

  return {
    messages,
    setMessages,
    sendMessage,
    decryptData,
    sendEncryptedMessage,
    sendTypingEvent,
    socketStatus,
    socketRef,
    setupWebSocketConnection,
    isTypingMessage,
  };
};

export { useMessagesAPI };

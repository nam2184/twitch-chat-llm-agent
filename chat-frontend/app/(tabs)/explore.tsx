import { StyleSheet, TouchableOpacity } from 'react-native';
import { GetUser } from "@/lib/kubb";
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import SimpleScrollView from '@/components/ui/normal-scroll-view';
import { useGetChats, useGetUser } from '@/lib/kubb';
import { Link, useRouter } from 'expo-router';


export default function ChatListScreen() {
  const { data } = useGetChats();
  const getUser = useGetUser();
  const currentUserId = getUser.data?.id
  const users = data?.users;
  const chats = data?.chats;
  return (
    <SimpleScrollView>
      <ThemedView style={styles.container}>
        <ThemedText type="title">Chats</ThemedText>

        {users?.map((user, index) => {
          const chat = chats?.find(
            (chat) =>
              (chat.user1_id === currentUserId && chat.user2_id === user.id) ||
              (chat.user2_id === currentUserId && chat.user1_id === user.id)
          );

          if (!chat) return null;

          const chatId = chat.id;
          const receiver : GetUser = {
            id : user.id,
            username: user.username,
            first_name : user.first_name,
            surname : user.surname,
            email : user.email,
          };

          return (
            <ThemedView key={index} style={styles.chatItem}>
              <ThemedText style={styles.chatText}>{user.username}</ThemedText>
                <Link
                    href={{
                      pathname: '/chat/[chatId]',
                      params: {
                        chatId: String(chatId),
                        sender: JSON.stringify(getUser.data),
                        receiver: JSON.stringify(receiver),
                      }
                    }}
                    asChild
                  >
                <TouchableOpacity style={styles.openButton}>
                  <ThemedText style={styles.openText}>Open</ThemedText>
                </TouchableOpacity>
              </Link>
            </ThemedView>
          );
        })}
      </ThemedView>
    </SimpleScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    gap: 20,
  },
  chatItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#ccc',
  },
  chatText: {
    fontSize: 16,
  },
  openButton: {
    backgroundColor: 'white',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  openText: {
    color: 'black',
    fontWeight: 'bold',
  },
});


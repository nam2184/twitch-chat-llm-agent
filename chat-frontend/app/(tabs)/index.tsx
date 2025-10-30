import { StyleSheet, TouchableOpacity } from 'react-native';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import SimpleScrollView from '@/components/ui/normal-scroll-view';
import { useGetChats, useGetUsers } from '@/lib/kubb';

export default function AddFriendScreen() {
  const getUsers = useGetUsers()
  const getChats = useGetChats()
  return (
    <SimpleScrollView>
      <ThemedView style={styles.container}>
        {/* Added Friends */}
        <ThemedView style={styles.section}>
          <ThemedText type="title">Your Friends</ThemedText>
          {getChats.data?.users?.map((friend, idx) => (
            <ThemedView key={idx} style={styles.friendItem}>
              <ThemedText>{friend.username}</ThemedText>
            </ThemedView>
          ))}
        </ThemedView>

        {/* Suggested Friends */}
        <ThemedView style={styles.section}>
          <ThemedText type="title">People You May Know</ThemedText>
          {getUsers.data?.array?.map((friend, idx) => (
            <ThemedView key={idx} style={styles.friendItem}>
              <ThemedText>{friend.username}</ThemedText>
              <TouchableOpacity style={styles.addButton}>
                <ThemedText style={styles.addText}>Add</ThemedText>
              </TouchableOpacity>
            </ThemedView>
          ))}
        </ThemedView>
      </ThemedView>
    </SimpleScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    gap: 30,
  },
  section: {
    gap: 12,
  },
  friendItem: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: "#ccc",
  },
  addButton: {
    backgroundColor: "white",
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  addText: {
    color: "black",
    fontWeight: "bold",
  },
  removeButton: {
    backgroundColor: "red",
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  removeText: {
    color: "white",
    fontWeight: "bold",
  },
});


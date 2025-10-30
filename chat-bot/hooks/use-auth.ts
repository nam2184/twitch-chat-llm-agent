import AsyncStorage from '@react-native-async-storage/async-storage';
import { PostAuthResponse } from "@/lib/kubb";

const loadCredentials = async (response: PostAuthResponse) => {
  try {
    console.log("Loading creds");
    await AsyncStorage.removeItem('access_token');
    await AsyncStorage.removeItem('refresh_token');
    await AsyncStorage.setItem('access_token', response.access_token!);
    await AsyncStorage.setItem('refresh_token', response.refresh_token!);
    return response;
  } catch (error) {
    console.error("Error signing in:", error);
    throw error;
  }
};

const signOut = async () => {
  await AsyncStorage.removeItem('access_token');
  await AsyncStorage.removeItem('refresh_token');
  // Instead of window.location.href, use navigation
};

export { loadCredentials, signOut}

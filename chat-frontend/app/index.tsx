import React from "react";
import { View, Text, TextInput, Button, StyleSheet, ActivityIndicator, TouchableOpacity } from "react-native";
import { Controller, useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { Stack, useRouter } from "expo-router";
import { usePostAuth } from "@/lib/kubb";
import { loadCredentials } from "@/hooks/use-auth";

const loginSchema = z.object({
  username: z.string().min(1, "Username is required"),
  password: z.string().min(1, "Password is required"),
});


export default function Login() {
  const router = useRouter();
  const [isLoading, setIsLoading] = React.useState(false);

  const form = useForm<z.infer<typeof loginSchema>>({
    resolver: zodResolver(loginSchema),
    defaultValues: { username: "", password: "" },
  });

  const postAuthMutation = usePostAuth();
  const onSubmit = React.useCallback(
      async ({ username, password }: z.infer<typeof loginSchema>) => {
        try {
          setIsLoading(true);
          const response = await postAuthMutation.mutateAsync({ 
            data : {
              username: username,
              password: password,
            }
          });
          console.log(response)
          await loadCredentials(response);
          setIsLoading(false);
        } catch (e: unknown) {
            if (e instanceof Error) {
              form.setError("root", {
                message: e.message,
              });
          }
        } finally {
          setIsLoading(false);
        }
        },
      [form, postAuthMutation],
  );
  React.useEffect(() => {
    if (postAuthMutation.data?.user) {
      router.replace("/(tabs)");
    }
  }, [router, postAuthMutation]);
  return (
    <View style={styles.container}>
      <Stack.Screen options={{ title: "Login" }} />

      {/* Username field */}
      <Controller
        control={form.control}
        name="username"
        render={({ field: { onChange, value } }) => (
          <TextInput
            style={styles.input}
            placeholder="Username"
            value={value}
            onChangeText={onChange}
            autoCapitalize="none"
          />
        )}
      />
      {/* Password field */}
      <Controller
        control={form.control}
        name="password"
        render={({ field: { onChange, value } }) => (
          <TextInput
            style={styles.input}
            placeholder="Password"
            value={value}
            onChangeText={onChange}
            secureTextEntry
          />
        )}
      />

      {/* Root error */}
      {form.formState.errors.root && (
        <Text style={styles.error}>{form.formState.errors.root.message}</Text>
      )}
      {/* Submit button */}
      {isLoading ? (
        <ActivityIndicator size="large" />
      ) : (
      <TouchableOpacity 
        style={styles.button} 
        onPress={form.handleSubmit(onSubmit)} 
        disabled={isLoading} // prevent double taps
      >
        <Text style={styles.buttonText}>
          {isLoading ? "Logging in..." : "Login"}
        </Text>
      </TouchableOpacity>      
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    padding: 20,
  },
  input: {
    borderWidth: 1,
    borderColor: "#ccc",
    padding: 12,
    marginBottom: 10,
    borderRadius: 8,
    color: "white", 
  },
  inputFocused: {
    borderWidth: 2, 
    borderColor: "#000", 
  },
  error: {
    color: "red",
    marginBottom: 10,
  },
  button: {
    backgroundColor: "white",
    padding: 15,
    borderRadius: 30,
    alignItems: "center",
    width: "60%",        
    alignSelf: "center",
    marginTop: 80,
  },
  buttonText: {
    color: "black",
    fontWeight: "bold",
  },
});


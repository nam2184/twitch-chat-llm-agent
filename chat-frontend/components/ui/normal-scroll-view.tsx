import type { PropsWithChildren } from 'react';
import { StyleSheet } from 'react-native';
import Animated from 'react-native-reanimated';

import { ThemedView } from '@/components/themed-view';
import { useThemeColor } from '@/hooks/use-theme-color';

type Props = PropsWithChildren<{}>;

export default function SimpleScrollView({ children }: Props) {
  const backgroundColor = useThemeColor({}, 'background');

  return (
    <Animated.ScrollView
      style={{ backgroundColor, flex: 1 }}
      scrollEventThrottle={16}
      showsVerticalScrollIndicator={false}
    >
      <ThemedView style={styles.content}>{children}</ThemedView>
    </Animated.ScrollView>
  );
}

const styles = StyleSheet.create({
  content: {
    flex: 1,
    padding: 32,
    gap: 16,
  },
});

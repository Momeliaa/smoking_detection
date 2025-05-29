import { useEffect } from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useFrameworkReady } from '@/hooks/useFrameworkReady';
import { registerForPushNotificationsAsync, setupNotificationListener, handleNotificationData } from './utils/notification-handler';
import { Alert } from 'react-native';

export default function RootLayout() {
  useFrameworkReady();

  useEffect(() => {
    let notificationSubscription: { remove: () => void } | null = null;

    async function setupNotifications() {
      try {
        const token = await registerForPushNotificationsAsync();
        if (token) {
          console.log('Push token:', token);
        }

        notificationSubscription = setupNotificationListener(async (notification) => {
          try {
            const data = notification.request.content.data;
            const pointData = await handleNotificationData(data);
            
            // Here you would typically update your app's state with the new point
            // For now, we'll just show an alert
            Alert.alert(
              'New Watch Point Update',
              `Status: ${pointData.status}\nLocation: ${pointData.address}`,
              [{ text: 'OK' }]
            );
          } catch (error) {
            console.error('Error processing notification:', error);
            Alert.alert(
              'Notification Error',
              'Failed to process notification data. Please try again.',
              [{ text: 'OK' }]
            );
          }
        });
      } catch (error) {
        console.error('Error setting up notifications:', error);
      }
    }

    setupNotifications();

    return () => {
      if (notificationSubscription) {
        notificationSubscription.remove();
      }
    };
  }, []);

  return (
    <>
      <Stack screenOptions={{ headerShown: false }}>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen name="+not-found" options={{ title: 'Oops!' }} />
      </Stack>
      <StatusBar style="auto" />
    </>
  );
}
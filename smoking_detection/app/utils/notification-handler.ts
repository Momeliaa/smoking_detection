import * as Notifications from 'expo-notifications';
import Constants from 'expo-constants';
import { Platform } from 'react-native';

// ===== 푸시 알림 설정 =====
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
  }),
});

// ===== 디바이스 푸시 알림 등록 =====
export async function registerForPushNotificationsAsync() {
  try {
    if (Platform.OS === 'web') {
      console.log('Push notifications are not supported on web platform');
      return null;
    }

    const { status: existingStatus } = await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;

    if (existingStatus !== 'granted') {
      const { status } = await Notifications.requestPermissionsAsync();
      finalStatus = status;
    }

    if (finalStatus !== 'granted') {
      console.log('Failed to get push token for push notification!');
      return null;
    }

    const token = await Notifications.getExpoPushTokenAsync({
      projectId: Constants.expoConfig?.extra?.expoProjectId,
    }).catch((err) => {
      console.error('Error getting push token:', err);
      return null;
    });

    if (token?.data) {
      console.log('✅ 푸시 토큰:', token.data);
    }

    return token;
  } catch (error) {
    console.error('Error in registerForPushNotificationsAsync:', error);
    return null;
  }
}

// ===== 알림 수신 리스너 =====
export function setupNotificationListener(callback: (notification: Notifications.Notification) => void) {
  try {
    const foregroundSubscription = Notifications.addNotificationReceivedListener(notification => {
      console.log('Received notification in foreground:', notification);
      callback(notification);
    });

    const responseSubscription = Notifications.addNotificationResponseReceivedListener(response => {
      console.log('Notification tapped:', response);
      callback(response.notification);
    });

    return {
      remove: () => {
        foregroundSubscription.remove();
        responseSubscription.remove();
      }
    };
  } catch (error) {
    console.error('Error in setupNotificationListener:', error);
    return {
      remove: () => {}
    };
  }
}

// ===== 알림 데이터 핸들링 =====
export async function handleNotificationData(data: any) {
  try {
    if (!data) throw new Error('No notification data received');

    return {
      id: Date.now().toString(), // fallback id
      latitude: 0,
      longitude: 0,
      address: data.address || 'Unknown Location',
      status: data.status,
      photo: data.photo,
      updatedAt: new Date().toISOString(),
      statusHistory: {
        yellow: data.status === 'yellow' ? 1 : 0,
        red: data.status === 'red' ? 1 : 0,
      }
    };
  } catch (error) {
    console.error('Error handling notification data:', error);
    throw error;
  }
}

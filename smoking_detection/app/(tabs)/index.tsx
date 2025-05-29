import { useEffect, useState, useRef } from 'react';
import { StyleSheet, View, Text, Image, Modal, TouchableOpacity, ScrollView } from 'react-native';
import MapView, { Marker, PROVIDER_GOOGLE } from 'react-native-maps';
import * as Location from 'expo-location';
import * as Notifications from 'expo-notifications';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface WatchPoint {
  id: string;
  latitude: number;
  longitude: number;
  address: string;
  status: 'green' | 'red';
  updatedAt: string;
  photo?: string;
  history?: { photo: string; timestamp: string }[];
}

export default function MapScreen() {
  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<WatchPoint | null>(null);
  const [watchPoints, setWatchPoints] = useState<WatchPoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [mapRefreshKey, setMapRefreshKey] = useState(0);
  const mapRef = useRef<MapView>(null);

  const storeHistory = async (address: string, history: { photo: string; timestamp: string }[]) => {
    await AsyncStorage.setItem(`history:${address}`, JSON.stringify(history));
  };

  const getHistory = async (address: string) => {
    const data = await AsyncStorage.getItem(`history:${address}`);
    return data ? JSON.parse(data) : [];
  };

  const openPoint = async (point: WatchPoint) => {
    const history = await getHistory(point.address);
    setSelectedPoint({ ...point, history });
  };

  useEffect(() => {
    (async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setError('Location permission not granted');
        return;
      }

      const locations = ['한성대학교', '한성대역', '한성대 공학관', '혜화역', '대학로'];
      const geocodedCoords = await Promise.all(locations.map((loc) => Location.geocodeAsync(loc)));
      const photoUrl = 'https://raw.githubusercontent.com/sperola37/test-repo-1743934500179/refs/heads/main/photo1.png';

      const generateRandomHistory = (): { photo: string; timestamp: string }[] => {
        const count = Math.floor(Math.random() * 5) + 1;
        const now = Date.now();
        return Array.from({ length: count }, () => {
          const offset = Math.floor(Math.random() * 7 * 24 * 60 * 60 * 1000);
          return { photo: photoUrl, timestamp: new Date(now - offset).toISOString() };
        });
      };

      const examplePoints: WatchPoint[] = geocodedCoords.map((coords, idx) => {
        const history = generateRandomHistory();
        const address = locations[idx];
        storeHistory(address, history);
        return {
          id: `${idx}`,
          latitude: coords[0].latitude,
          longitude: coords[0].longitude,
          address,
          status: 'green',
          updatedAt: new Date().toISOString(),
          photo: photoUrl,
          history,
        };
      });

      setLocation({
        coords: {
          latitude: geocodedCoords[0][0].latitude,
          longitude: geocodedCoords[0][0].longitude,
          accuracy: 5,
          altitude: 0,
          altitudeAccuracy: 5,
          heading: 0,
          speed: 0,
        },
        timestamp: Date.now(),
      });

      setWatchPoints(examplePoints);
    })();
  }, []);

  useEffect(() => {
    const subscription = Notifications.addNotificationReceivedListener(async (notification) => {
      const data = notification.request.content.data;
      if (!data?.address || !data?.status) return;

      const coords = await Location.geocodeAsync(data.address);
      if (!coords.length) return;
      const { latitude, longitude } = coords[0];
      const newTimestamp = new Date().toISOString();

      setWatchPoints((prev) => {
        const updated = prev.map((p) => {
          if (p.address === data.address) {
            const newHistory = data.status === 'smoking'
              ? [...(p.history || []), { photo: data.photo, timestamp: newTimestamp }]
              : p.history;

            if (data.status === 'smoking') {
              storeHistory(p.address, newHistory || []);
            }

            return {
              ...p,
              status: (data.status === 'smoking' ? 'red' : 'green') as 'red' | 'green',
              updatedAt: newTimestamp,
              photo: data.photo,
              history: newHistory,
            };
          }
          return p;
        });

        const exists = prev.some((p) => p.address === data.address);
        if (!exists) {
          const newPoint: WatchPoint = {
            id: Date.now().toString(),
            latitude,
            longitude,
            address: data.address,
            status: data.status === 'smoking' ? 'red' : 'green',
            updatedAt: newTimestamp,
            photo: data.photo,
            history: data.status === 'smoking'
              ? [{ photo: data.photo, timestamp: newTimestamp }]
              : [],
          };
          if (data.status === 'smoking') {
            storeHistory(newPoint.address, newPoint.history || []);
          }
          return [...updated, newPoint];
        }

        return [...updated];
      });

      setMapRefreshKey((prev) => prev + 1); // 🔄 강제 리렌더링
    });

    return () => subscription.remove();
  }, []);

  if (error) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>{error}</Text>
      </View>
    );
  }

  if (!location) {
    return (
      <View style={styles.container}>
        <Text>Loading location...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <MapView
        key={mapRefreshKey} // 🔄 키 변경 시 MapView 재생성
        ref={mapRef}
        provider={PROVIDER_GOOGLE}
        style={styles.map}
        initialRegion={{
          latitude: location.coords.latitude,
          longitude: location.coords.longitude,
          latitudeDelta: 0.05,
          longitudeDelta: 0.05,
        }}
      >
        {watchPoints.map((point) => (
          <Marker
            key={point.address}
            coordinate={{ latitude: point.latitude, longitude: point.longitude }}
            pinColor={point.status === 'green' ? '#00cc00' : '#ff0000'}
            onPress={() => openPoint(point)}
          />
        ))}
      </MapView>

      {/* 상세 모달 */}
      <Modal visible={!!selectedPoint} transparent animationType="slide" onRequestClose={() => setSelectedPoint(null)}>
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>{selectedPoint?.address}</Text>
            <Text>
              상태:{' '}
              <Text style={{ color: selectedPoint?.status === 'green' ? 'green' : 'red' }}>
                {selectedPoint?.status === 'green' ? '흡연 감지 안됨' : '흡연 감지됨'}
              </Text>
            </Text>
            <Text>갱신 시각: {new Date(selectedPoint?.updatedAt || '').toLocaleString()}</Text>
            {selectedPoint?.photo && (
              <TouchableOpacity onPress={() => setShowHistoryModal(true)}>
                <Image source={{ uri: selectedPoint.photo }} style={styles.modalImage} />
              </TouchableOpacity>
            )}
            <TouchableOpacity style={styles.closeButton} onPress={() => setSelectedPoint(null)}>
              <Text style={styles.closeButtonText}>닫기</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* 히스토리 모달 */}
      <Modal visible={showHistoryModal} transparent animationType="slide" onRequestClose={() => setShowHistoryModal(false)}>
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>과거 감지 사진</Text>
            <ScrollView horizontal>
              {selectedPoint?.history?.map((item, index) => (
                <View key={index} style={{ marginRight: 10 }}>
                  <Image source={{ uri: item.photo }} style={{ width: 100, height: 100, borderRadius: 8 }} />
                  <Text style={{ fontSize: 12 }}>
                    {new Date(item.timestamp).toLocaleString('ko-KR', {
                      year: 'numeric',
                      month: '2-digit',
                      day: '2-digit',
                      hour: '2-digit',
                      minute: '2-digit',
                      hour12: false,
                    })}
                  </Text>
                </View>
              ))}
            </ScrollView>
            <TouchableOpacity style={styles.closeButton} onPress={() => setShowHistoryModal(false)}>
              <Text style={styles.closeButtonText}>닫기</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  map: { flex: 1 },
  modalContainer: {
    flex: 1,
    justifyContent: 'flex-end',
    backgroundColor: 'rgba(0,0,0,0.4)',
  },
  modalContent: {
    backgroundColor: '#fff',
    padding: 20,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  modalImage: {
    width: '100%',
    height: 200,
    marginTop: 10,
    borderRadius: 8,
  },
  closeButton: {
    marginTop: 15,
    backgroundColor: '#007AFF',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  closeButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  errorText: {
    marginTop: 50,
    fontSize: 16,
    color: 'red',
    textAlign: 'center',
  },
});

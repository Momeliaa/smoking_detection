import { useState, useCallback } from 'react';
import { StyleSheet, View, Text, Dimensions, ScrollView, Modal, TouchableOpacity, Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';
import {
  VictoryBar,
  VictoryChart,
  VictoryTheme,
  VictoryAxis,
  VictoryGroup,
  VictoryScatter,
  VictoryLabel,
} from 'victory-native';
import * as Notifications from 'expo-notifications';

const weekdayLabels = ['일', '월', '화', '수', '목', '금', '토'];

export default function StatisticsScreen() {
  const [chartData, setChartData] = useState<{ point: string; count: number }[]>([]);
  const [timeSeriesRed, setTimeSeriesRed] = useState<{ time: string; yValue: number }[]>([]);
  const [chartDataWeekday, setChartDataWeekday] = useState<{ weekday: string; count: number }[]>([]);
  const [pointIndexMap, setPointIndexMap] = useState<Record<string, number>>({});
  const [tokenModalVisible, setTokenModalVisible] = useState(false);
  const [expoToken, setExpoToken] = useState('');

  const loadHistory = async () => {
    const keys = await AsyncStorage.getAllKeys();
    const watchKeys = keys.filter((k) => k.startsWith('history:'));
    const allEntries = await AsyncStorage.multiGet(watchKeys);

    const pointCount: Record<string, number> = {};
    const rawTimeData: { time: string; point: string }[] = [];
    const dayCount: number[] = new Array(7).fill(0);

    const oneWeekAgo = new Date();
    oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);

    for (const [key, value] of allEntries) {
      const address = key.replace('history:', '');
      const entries = value ? JSON.parse(value) : [];

      const recentEntries = entries.filter((entry: { timestamp: string }) => {
        const ts = new Date(entry.timestamp);
        return ts >= oneWeekAgo;
      });

      pointCount[address] = recentEntries.length;

      recentEntries.forEach((entry: { timestamp: string }) => {
        const date = new Date(entry.timestamp);
        const hour = date.getHours();
        const weekday = date.getDay();
        rawTimeData.push({ time: String(hour), point: address });
        dayCount[weekday]++;
      });
    }

    const pointList = Object.keys(pointCount);
    const pointMap: Record<string, number> = {};
    pointList.forEach((point, idx) => {
      pointMap[point] = idx + 1;
    });

    const timeDataWithNumericY = rawTimeData.map((entry) => ({
      time: entry.time,
      yValue: pointMap[entry.point],
    }));

    const weekdayData = [1, 2, 3, 4, 5, 6, 0].map((dayIdx) => ({
      weekday: weekdayLabels[dayIdx],
      count: dayCount[dayIdx],
    }));

    setPointIndexMap(pointMap);
    setChartData(pointList.map((point) => ({ point, count: pointCount[point] })));
    setTimeSeriesRed(timeDataWithNumericY);
    setChartDataWeekday(weekdayData);
  };

  useFocusEffect(
    useCallback(() => {
      loadHistory();
    }, [])
  );

  const sendTokenToEC2 = async () => {
    try {
      const token = await Notifications.getExpoPushTokenAsync();
      setExpoToken(token.data);

      const response = await fetch('http://43.200.193.228:5000/register-token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: token.data })
      });

      const result = await response.json();
      if (result.success) {
        Alert.alert('성공', 'Expo 푸시 토큰이 EC2로 전송되었습니다.');
      } else {
        Alert.alert('실패', result.message || '토큰 전송 실패');
      }
    } catch (error) {
      console.error('토큰 전송 오류:', error);
      Alert.alert('에러', '토큰 전송 중 오류 발생');
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>통계 자료</Text>
        <TouchableOpacity style={styles.tokenButton} onPress={sendTokenToEC2}>
          <Text style={styles.tokenButtonText}>토큰 전송</Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.graphTitle}>지난 주 장소 별 흡연 횟수</Text>
      <VictoryChart
        theme={VictoryTheme.material}
        width={Dimensions.get('window').width - 40}
        height={300}
        domainPadding={{ x: 50 }}
        padding={{ top: 10, bottom: 50, left: 70, right: 20 }}
        categories={{ x: chartData.map((item) => item.point) }}
      >
        <VictoryAxis
          label="장소 (address)"
          axisLabelComponent={<VictoryLabel dy={25} style={{ fontSize: 12 }} />}
          tickFormat={(t) => String(t)}
          style={{ tickLabels: { fontSize: 9, textAnchor: 'middle', padding: 10 } }}
        />
        <VictoryAxis
          dependentAxis
          tickFormat={(x) => Math.round(x)}
          style={{ tickLabels: { fontSize: 12, padding: 5 } }}
        />
        <VictoryGroup offset={25}>
          <VictoryBar
            data={chartData}
            x="point"
            y="count"
            style={{ data: { fill: '#FF0000' } }}
          />
        </VictoryGroup>
      </VictoryChart>

      <Text style={styles.graphTitle}>지난 주 시간대 별 흡연 횟수</Text>
      <VictoryChart
        theme={VictoryTheme.material}
        width={Dimensions.get('window').width - 40}
        height={280}
        domainPadding={{ x: 20, y: 10 }}
        padding={{ top: 10, bottom: 40, left: 70, right: 20 }}
        categories={{ x: Array.from({ length: 25 }, (_, i) => String(i)) }}
      >
        <VictoryAxis
          label="시간 (시)"
          tickValues={["0", "4", "8", "12", "16", "20", "24"]}
          tickFormat={(t) => t}
          style={{
            axisLabel: { padding: 25, fontSize: 12 },
            tickLabels: { fontSize: 10 },
          }}
        />
        <VictoryAxis
          dependentAxis
          tickValues={Object.values(pointIndexMap)}
          tickFormat={(val) => Object.entries(pointIndexMap).find(([_, v]) => v === val)?.[0] ?? ''}
          style={{ tickLabels: { fontSize: 10, padding: 2 } }}
        />
        <VictoryScatter
          data={timeSeriesRed}
          x="time"
          y="yValue"
          size={3}
          style={{ data: { fill: '#FF0000' } }}
        />
      </VictoryChart>

      <Text style={styles.graphTitle}>지난 주 요일 별 흡연 횟수</Text>
      <VictoryChart
        theme={VictoryTheme.material}
        width={Dimensions.get('window').width - 40}
        height={280}
        domainPadding={{ x: 30 }}
        padding={{ top: 10, bottom: 40, left: 70, right: 20 }}
        categories={{ x: ['월', '화', '수', '목', '금', '토', '일'] }}
      >
        <VictoryAxis
          label="요일"
          tickFormat={(t) => t}
          style={{
            axisLabel: { padding: 25, fontSize: 12 },
            tickLabels: { fontSize: 10 },
          }}
        />
        <VictoryAxis
          dependentAxis
          tickFormat={(x) => Math.round(x)}
          style={{ tickLabels: { fontSize: 12, padding: 5 } }}
        />
        <VictoryBar
          data={chartDataWeekday}
          x="weekday"
          y="count"
          style={{ data: { fill: '#FF0000' } }}
        />
      </VictoryChart>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  header: {
    paddingTop: 40,
    padding: 20,
    backgroundColor: '#f8f9fa',
    borderBottomWidth: 1,
    borderBottomColor: '#e9ecef',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#212529',
  },
  tokenButton: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    backgroundColor: '#007AFF',
    borderRadius: 8,
  },
  tokenButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  graphTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 20,
    marginTop: 20,
    marginBottom: 5,
    color: '#343a40',
  },
  modalOverlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  modalContainer: {
    width: '80%',
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 20,
    alignItems: 'center',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  modalText: {
    fontSize: 12,
    textAlign: 'center',
    marginBottom: 20,
  },
  closeButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 10,
    paddingHorizontal: 30,
    borderRadius: 8,
  },
  closeButtonText: {
    color: '#fff',
    fontWeight: 'bold',
  },
});

import { useForecast } from "@/hooks/useForecasts"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts"

interface Props {
  stationId: string
  stationName?: string
}

export default function ForecastChart({ stationId, stationName }: Props) {
  const { data, isLoading } = useForecast(stationId)
  if (isLoading)
    return (
      <Card>
        <CardContent>
          <p className="text-gray-400">Loading forecast...</p>
        </CardContent>
      </Card>
    )
  const forecast = data?.forecast ?? []
  if (forecast.length === 0) return null

  const chartData = forecast.map(
    (f: { timestamp: string; predicted: number; confidence: number }) => ({
      hour: new Date(f.timestamp).getHours(),
      predicted: f.predicted,
      confidence: Math.round(f.confidence * 100),
    })
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">
          24h Forecast — {stationName ?? stationId}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="hour"
              tick={{ fontSize: 11 }}
              label={{
                value: "Hour",
                position: "insideBottom",
                offset: -4,
                fontSize: 11,
              }}
            />
            <YAxis
              tick={{ fontSize: 11 }}
              label={{
                value: "Passengers",
                angle: -90,
                position: "insideLeft",
                fontSize: 11,
              }}
            />
            <Tooltip
              formatter={(value, name) => [
                name === "predicted" ? `${value} pax` : `${value}%`,
                name === "predicted" ? "Predicted" : "Confidence",
              ]}
              labelFormatter={(label) => `${String(label).padStart(2, "0")}:00`}
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="confidence"
              stroke="#94a3b8"
              strokeWidth={1}
              strokeDasharray="4 4"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

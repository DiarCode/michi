import { useForecast } from "@/hooks/useForecasts";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import ConfidenceBadge from "@/components/ui/ConfidenceBadge";
import {
  Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Area, ComposedChart,
} from "recharts";

interface Props { stationId: string; stationName?: string }

export default function ForecastChart({ stationId, stationName }: Props) {
  const { data, isLoading } = useForecast(stationId);
  if (isLoading) return <Card><CardContent><p className="text-muted-foreground">Loading forecast...</p></CardContent></Card>;
  const forecast = data?.forecast ?? [];
  if (forecast.length === 0) return null;

  // Average confidence for the badge
  const avgConfidence = forecast.reduce((sum: number, f: { confidence: number }) => sum + f.confidence, 0) / forecast.length;
  const modelVersion = (data as { model_version?: string })?.model_version;

  const chartData = forecast.map((f: { timestamp: string; predicted: number; confidence: number }) => {
    const predicted = f.predicted;
    const margin = predicted * (1 - f.confidence);
    return {
      hour: new Date(f.timestamp).getHours(),
      predicted,
      upper: Math.round(predicted + margin),
      lower: Math.round(predicted - margin),
      confidence: Math.round(f.confidence * 100),
    };
  });

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm">24h Forecast — {stationName ?? stationId}</CardTitle>
        <ConfidenceBadge confidence={avgConfidence} modelVersion={modelVersion} compact />
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="hour" tick={{ fontSize: 11 }} label={{ value: "Hour", position: "insideBottom", offset: -4, fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} label={{ value: "Passengers", angle: -90, position: "insideLeft", fontSize: 11 }} />
            <Tooltip
              formatter={(value: number | string, name: string) => {
                if (name === "predicted") return [`${value} passengers`, "Predicted"];
                if (name === "confidence") return [`${value}%`, "Confidence"];
                return [String(value), name];
              }}
              labelFormatter={(label: number) => `${String(label).padStart(2, "0")}:00`}
            />
            {/* Confidence band as shaded area */}
            <Area
              type="monotone"
              dataKey="upper"
              stroke="none"
              fill="var(--primary)"
              fillOpacity={0.08}
            />
            <Area
              type="monotone"
              dataKey="lower"
              stroke="none"
              fill="var(--background)"
              fillOpacity={0.8}
            />
            <Line type="monotone" dataKey="predicted" stroke="var(--primary)" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="confidence" stroke="var(--muted-foreground)" strokeWidth={1} strokeDasharray="4 4" dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
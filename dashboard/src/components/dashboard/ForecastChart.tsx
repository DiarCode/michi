import { useForecast } from "@/hooks/useForecasts";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

interface Props { stationId: string; stationName?: string }

export default function ForecastChart({ stationId, stationName }: Props) {
  const { data, isLoading } = useForecast(stationId);
  if (isLoading) return <Card><CardContent><p className="text-gray-400">Loading forecast...</p></CardContent></Card>;
  const forecast = data?.forecast ?? [];
  const maxVal = Math.max(...forecast.map((f: { predicted: number }) => f.predicted), 1);

  return (
    <Card>
      <CardHeader><CardTitle className="text-sm">24h Forecast — {stationName ?? stationId}</CardTitle></CardHeader>
      <CardContent>
        <div className="flex items-end gap-1 h-32">
          {forecast.map((f: { timestamp: string; predicted: number; confidence: number }, i: number) => (
            <div key={i} className="flex flex-col items-center flex-1">
              <div className="w-full bg-blue-500 rounded-t" style={{ height: `${(f.predicted / maxVal) * 100}%` }} title={`${f.predicted} passengers · ${Math.round(f.confidence * 100)}% confidence`} />
              <span className="text-[9px] text-gray-400 mt-1">{new Date(f.timestamp).getHours()}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

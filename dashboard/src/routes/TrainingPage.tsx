import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchTrainingStatus, startTraining, uploadRidership } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Upload, Play, Activity } from "lucide-react";

export default function TrainingPage() {
  const qc = useQueryClient();
  const [epochs, setEpochs] = useState(50);
  const [file, setFile] = useState<File | null>(null);
  const [uploadResult, setUploadResult] = useState<{ rows_received: number; filename: string } | null>(null);

  const { data: status } = useQuery({ queryKey: ["training"], queryFn: fetchTrainingStatus, refetchInterval: 5000 });

  const trainMut = useMutation({
    mutationFn: () => startTraining(epochs),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["training"] }),
  });

  const uploadMut = useMutation({
    mutationFn: () => file ? uploadRidership(file) : Promise.reject("No file"),
    onSuccess: (data) => { setUploadResult(data); setFile(null); },
  });

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">Model Training</h2>
      <p className="text-sm text-gray-500 dark:text-gray-400">Manage DTS-GSSF model training, monitor status, and upload ridership data.</p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-500 dark:text-gray-400">Status</CardTitle>
            <Activity className="h-4 w-4 text-gray-400" />
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold capitalize">{status?.status ?? "—"}</p>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{status?.model_version ?? "—"}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-500 dark:text-gray-400">Epochs Trained</CardTitle>
          </CardHeader>
          <CardContent><p className="text-2xl font-bold">{status?.epochs_trained ?? "—"}</p></CardContent>
        </Card>

        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-500 dark:text-gray-400">Training Time</CardTitle>
          </CardHeader>
          <CardContent><p className="text-2xl font-bold">{status?.training_time_seconds ? `${status.training_time_seconds}s` : "—"}</p></CardContent>
        </Card>
      </div>

      {status?.metrics && (
        <Card>
          <CardHeader><CardTitle>Model Metrics</CardTitle></CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400 uppercase">MAE</p>
                <p className="text-xl font-bold">{status.metrics.mae}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400 uppercase">RMSE</p>
                <p className="text-xl font-bold">{status.metrics.rmse}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400 uppercase">MAPE</p>
                <p className="text-xl font-bold">{status.metrics.mape}%</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader><CardTitle>Start Training</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Epochs: {epochs}</label>
              <input type="range" min={1} max={500} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} className="w-full" />
              <div className="flex justify-between text-xs text-gray-400"><span>1</span><span>500</span></div>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Estimated time: {epochs * 7}s (~{Math.round(epochs * 7 / 60)}min)</p>
            <Button onClick={() => trainMut.mutate()} disabled={trainMut.isPending} className="w-full">
              <Play className="h-4 w-4 mr-2" />
              {trainMut.isPending ? "Starting..." : "Start Training"}
            </Button>
            {trainMut.data && (
              <p className="text-sm text-green-600 dark:text-green-400">
                Training started — {trainMut.data.epochs} epochs, version {trainMut.data.model_version}
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Upload Ridership Data</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <p className="text-xs text-gray-500 dark:text-gray-400">Upload a CSV with columns: station_id, timestamp, passengers</p>
            <input
              type="file" accept=".csv"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="block w-full text-sm text-gray-500 dark:text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:bg-blue-50 dark:file:bg-blue-900/30 file:text-blue-700 dark:file:text-blue-300 hover:file:bg-blue-100 dark:hover:file:bg-blue-900/50"
            />
            <Button onClick={() => uploadMut.mutate()} disabled={!file || uploadMut.isPending} className="w-full">
              <Upload className="h-4 w-4 mr-2" />
              {uploadMut.isPending ? "Uploading..." : "Upload CSV"}
            </Button>
            {uploadResult && (
              <p className="text-sm text-green-600 dark:text-green-400">
                Uploaded {uploadResult.rows_received} rows from {uploadResult.filename}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {status && (
        <p className="text-xs text-gray-400 dark:text-gray-500">
          Last trained: {new Date(status.last_trained).toLocaleString()} · Version: {status.model_version}
        </p>
      )}
    </div>
  );
}
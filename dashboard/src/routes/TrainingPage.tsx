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
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-extrabold text-michi-dark">Model Training</h1>
        <p className="text-base text-michi-muted mt-1">Manage DTS-GSSF model training, monitor status, and upload ridership data</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <Card>
          <CardContent className="p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-michi-muted font-medium">Status</span>
              <div className="w-8 h-8 rounded-full bg-michi-lime/15 flex items-center justify-center">
                <Activity size={16} className="text-michi-lime-dark" />
              </div>
            </div>
            <p className="text-3xl font-extrabold text-michi-dark capitalize">{status?.status ?? "—"}</p>
            <p className="text-sm text-michi-muted mt-1">{status?.model_version ?? "—"}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-5">
            <span className="text-sm text-michi-muted font-medium">Epochs Trained</span>
            <p className="text-3xl font-extrabold text-michi-dark mt-2">{status?.epochs_trained ?? "—"}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-5">
            <span className="text-sm text-michi-muted font-medium">Training Time</span>
            <p className="text-3xl font-extrabold text-michi-dark mt-2">{status?.training_time_seconds ? `${status.training_time_seconds}s` : "—"}</p>
          </CardContent>
        </Card>
      </div>

      {status?.metrics && (
        <Card>
          <CardHeader><CardTitle>Model Metrics</CardTitle></CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-6 text-center">
              <div>
                <p className="text-sm text-michi-muted font-medium uppercase">MAE</p>
                <p className="text-3xl font-extrabold text-michi-dark mt-2">{status.metrics.mae}</p>
              </div>
              <div>
                <p className="text-sm text-michi-muted font-medium uppercase">RMSE</p>
                <p className="text-3xl font-extrabold text-michi-dark mt-2">{status.metrics.rmse}</p>
              </div>
              <div>
                <p className="text-sm text-michi-muted font-medium uppercase">MAPE</p>
                <p className="text-3xl font-extrabold text-michi-dark mt-2">{status.metrics.mape}%</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader><CardTitle>Start Training</CardTitle></CardHeader>
          <CardContent className="space-y-5">
            <div>
              <label className="block text-sm font-semibold text-michi-dark mb-2">Epochs: {epochs}</label>
              <input type="range" min={1} max={500} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} className="w-full accent-michi-lime" />
              <div className="flex justify-between text-xs text-michi-muted font-medium mt-1"><span>1</span><span>500</span></div>
            </div>
            <p className="text-sm text-michi-muted font-medium">Estimated time: {epochs * 7}s (~{Math.round(epochs * 7 / 60)}min)</p>
            <Button onClick={() => trainMut.mutate()} disabled={trainMut.isPending} variant="lime" className="w-full">
              <Play size={16} className="mr-1.5" />
              {trainMut.isPending ? "Starting..." : "Start Training"}
            </Button>
            {trainMut.data && (
              <p className="text-sm text-michi-lime-dark font-semibold">
                Training started — {trainMut.data.epochs} epochs, version {trainMut.data.model_version}
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Upload Ridership Data</CardTitle></CardHeader>
          <CardContent className="space-y-5">
            <p className="text-sm text-michi-muted font-medium">Upload a CSV with columns: station_id, timestamp, passengers</p>
            <input
              type="file" accept=".csv"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="block w-full text-sm text-michi-body file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-michi-lime/15 file:text-michi-lime-dark hover:file:bg-michi-lime/25"
            />
            <Button onClick={() => uploadMut.mutate()} disabled={!file || uploadMut.isPending} className="w-full">
              <Upload size={16} className="mr-1.5" />
              {uploadMut.isPending ? "Uploading..." : "Upload CSV"}
            </Button>
            {uploadResult && (
              <p className="text-sm text-michi-lime-dark font-semibold">
                Uploaded {uploadResult.rows_received} rows from {uploadResult.filename}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {status && (
        <p className="text-sm text-michi-muted font-medium">
          Last trained: {new Date(status.last_trained).toLocaleString()} · Version: {status.model_version}
        </p>
      )}
    </div>
  );
}
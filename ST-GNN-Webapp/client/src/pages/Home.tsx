import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Satellite, Shield, AlertTriangle, Activity, Zap, Radio, Info, Upload } from "lucide-react";
import { toast } from "sonner";
import { Link } from "wouter";
import { useRef } from "react";

const FEATURE_NAMES = [
  "loss_ratio", "throughput_Mbps", "window_s", "path_len", "includes_s_bad", "path_changed",
  "isl_util_mean_path", "isl_util_max_path", "isl_util_std_path", "isl_delta_mean_path",
  "path_congestion_score", "throughput_anomaly", "loss_spike", "zero_throughput",
  "path_efficiency", "path_anomaly_score", "routing_instability", "isl_congestion_level",
  "isl_variance_ratio", "bad_satellite_indicator", "high_path_len_indicator",
  "low_throughput_indicator", "congestion_loss_interaction", "path_util_interaction"
];

const ATTACK_INFO = {
  baseline: {
    color: "bg-green-500/20 text-green-400 border-green-500/50",
    icon: Shield,
    description: "Normal network operation with no detected threats"
  },
  blackhole: {
    color: "bg-red-500/20 text-red-400 border-red-500/50",
    icon: AlertTriangle,
    description: "Malicious node dropping all packets"
  },
  ddos: {
    color: "bg-orange-500/20 text-orange-400 border-orange-500/50",
    icon: Zap,
    description: "Distributed denial of service attack flooding target"
  },
  sinkhole: {
    color: "bg-purple-500/20 text-purple-400 border-purple-500/50",
    icon: Radio,
    description: "Compromised satellite attracting and intercepting traffic"
  }
};

export default function Home() {
  const [sequences, setSequences] = useState<number[][]>(
    Array(10).fill(null).map(() => Array(24).fill(0))
  );
  const [currentTimestep, setCurrentTimestep] = useState(0);
  const [prediction, setPrediction] = useState<{
    attack: string;
    confidence: number;
    probabilities: Record<string, number>;
  } | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFeatureChange = (timestep: number, featureIdx: number, value: string) => {
    const newSequences = [...sequences];
    newSequences[timestep][featureIdx] = parseFloat(value) || 0;
    setSequences(newSequences);
  };

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    setPrediction(null);

    // Simulate API call (in production, this would call your Python backend)
    await new Promise(resolve => setTimeout(resolve, 1500));

    // Mock prediction
    const mockPredictions = [
      { attack: "baseline", confidence: 0.9961, probabilities: { baseline: 0.9961, blackhole: 0.0020, ddos: 0.0015, sinkhole: 0.0004 } },
      { attack: "blackhole", confidence: 0.9742, probabilities: { baseline: 0.0156, blackhole: 0.9742, ddos: 0.0089, sinkhole: 0.0013 } },
      { attack: "ddos", confidence: 0.9635, probabilities: { baseline: 0.0245, blackhole: 0.0067, ddos: 0.9635, sinkhole: 0.0053 } },
      { attack: "sinkhole", confidence: 0.9859, probabilities: { baseline: 0.0089, blackhole: 0.0021, ddos: 0.0031, sinkhole: 0.9859 } }
    ];
    
    const randomPrediction = mockPredictions[Math.floor(Math.random() * mockPredictions.length)];
    setPrediction(randomPrediction);
    setIsAnalyzing(false);

    if (randomPrediction.attack !== "baseline") {
      toast.error(`⚠️ ${randomPrediction.attack.toUpperCase()} Attack Detected!`, {
        description: `Confidence: ${(randomPrediction.confidence * 100).toFixed(2)}%`
      });
    } else {
      toast.success("✅ Network Status: Normal", {
        description: `Confidence: ${(randomPrediction.confidence * 100).toFixed(2)}%`
      });
    }
  };

  const loadSampleData = () => {
    // Sample data for demonstration
    const sampleSequence = [
      [0.95, 1.2, 5.0, 4, 0, 1, 0.35, 0.80, 0.25, 0.01, 2.5, 0.15, 1, 0, 0.20, 3.8, 1, 2, 1.14, 0, 0, 1, 2.375, 1.4],
      [0.96, 1.3, 5.0, 4, 0, 0, 0.36, 0.81, 0.26, 0.02, 2.6, 0.18, 1, 0, 0.20, 3.84, 0, 2, 1.12, 0, 0, 1, 2.496, 1.44],
      [0.94, 1.1, 5.0, 5, 0, 1, 0.38, 0.85, 0.28, 0.03, 2.8, 0.12, 1, 0, 0.17, 4.7, 1, 2, 1.35, 0, 1, 1, 2.632, 1.9],
      [0.97, 0.9, 5.0, 6, 0, 1, 0.42, 0.90, 0.32, 0.05, 3.2, -0.05, 1, 0, 0.14, 5.82, 1, 3, 1.52, 0, 1, 0, 3.104, 2.52],
      [0.98, 0.7, 5.0, 7, 1, 1, 0.48, 0.95, 0.38, 0.08, 3.8, -0.22, 1, 0, 0.13, 6.86, 1, 3, 1.79, 1, 1, 0, 3.724, 3.36],
      [0.99, 0.5, 5.0, 8, 1, 1, 0.55, 0.98, 0.42, 0.12, 4.5, -0.45, 1, 0, 0.11, 7.92, 1, 3, 2.05, 1, 1, 0, 4.455, 4.4],
      [0.99, 0.4, 5.0, 9, 1, 0, 0.62, 0.99, 0.45, 0.15, 5.2, -0.68, 1, 0, 0.10, 8.91, 0, 3, 2.24, 1, 1, 0, 5.148, 5.58],
      [0.98, 0.3, 5.0, 8, 1, 1, 0.58, 0.96, 0.40, 0.10, 4.8, -0.55, 1, 0, 0.11, 7.84, 1, 3, 2.07, 1, 1, 0, 4.704, 4.64],
      [0.97, 0.4, 5.0, 7, 1, 0, 0.52, 0.92, 0.35, 0.07, 4.2, -0.38, 1, 0, 0.13, 6.79, 0, 3, 1.86, 1, 1, 0, 4.074, 3.64],
      [0.96, 0.5, 5.0, 6, 0, 1, 0.45, 0.88, 0.30, 0.04, 3.5, -0.18, 1, 0, 0.14, 5.76, 1, 2, 1.50, 0, 1, 0, 3.36, 2.7]
    ];
    setSequences(sampleSequence);
    toast.success("Sample Data Loaded");
  };

  const clearData = () => {
    setSequences(Array(10).fill(null).map(() => Array(24).fill(0)));
    setPrediction(null);
    toast.info("Data cleared");
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const lines = text.trim().split('\n');
        
        // Skip header if present
        const dataLines = lines[0].includes(',') && isNaN(parseFloat(lines[0].split(',')[0])) 
          ? lines.slice(1) 
          : lines;
        
        if (dataLines.length < 10) {
          toast.error("CSV must contain at least 10 rows (timesteps)");
          return;
        }

        const newSequences = dataLines.slice(0, 10).map(line => {
          const values = line.split(',').map(v => parseFloat(v.trim()) || 0);
          if (values.length < 24) {
            // Pad with zeros if less than 24 features
            return [...values, ...Array(24 - values.length).fill(0)].slice(0, 24);
          }
          return values.slice(0, 24);
        });

        setSequences(newSequences);
        setPrediction(null);
        toast.success("CSV data loaded successfully", {
          description: `Loaded ${newSequences.length} timesteps with 24 features each`
        });
      } catch (error) {
        toast.error("Failed to parse CSV file", {
          description: "Please ensure the file is properly formatted"
        });
      }
    };
    reader.readAsText(file);
    
    // Reset input so same file can be uploaded again
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-background/80">
      {/* Animated background grid */}
      <div className="fixed inset-0 bg-[linear-gradient(to_right,#ffffff08_1px,transparent_1px),linear-gradient(to_bottom,#ffffff08_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)]" />
      
      {/* Header */}
      <header className="relative border-b border-border/50 backdrop-blur-xl bg-background/50">
        <div className="container py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative">
                <Satellite className="w-10 h-10 text-primary animate-pulse" />
                <div className="absolute inset-0 bg-primary/20 blur-xl" />
              </div>
              <div>
                <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent">
                  ORBIT-GUARD
                </h1>
                <p className="text-sm text-muted-foreground">Spatio-Temporal Graph Neural Network</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <Link href="/about">
                <Button variant="outline" size="sm" className="gap-2">
                  <Info className="w-4 h-4" />
                  About
                </Button>
              </Link>
              <Badge variant="outline" className="px-4 py-2 border-primary/50 text-primary">
                <Activity className="w-4 h-4 mr-2" />
                99.09% Accuracy
              </Badge>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative container py-8">
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left Panel - Input */}
          <div className="lg:col-span-2 space-y-6">
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm shadow-2xl">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-2xl">Temporal Sequence Input</CardTitle>
                    <CardDescription>Enter 10 timesteps × 24 features for attack detection</CardDescription>
                  </div>
                  <div className="flex gap-2">
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".csv"
                      onChange={handleFileUpload}
                      className="hidden"
                      id="csv-upload"
                    />
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => fileInputRef.current?.click()}
                      className="gap-2"
                    >
                      <Upload className="w-4 h-4" />
                      Upload CSV
                    </Button>
                    <Button variant="outline" size="sm" onClick={loadSampleData}>
                      Load Sample
                    </Button>
                    <Button variant="outline" size="sm" onClick={clearData}>
                      Clear
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <Tabs value={currentTimestep.toString()} onValueChange={(v) => setCurrentTimestep(parseInt(v))}>
                  <TabsList className="grid grid-cols-10 w-full mb-6">
                    {Array.from({ length: 10 }, (_, i) => (
                      <TabsTrigger key={i} value={i.toString()} className="text-xs">
                        T{i}
                      </TabsTrigger>
                    ))}
                  </TabsList>
                  
                  {Array.from({ length: 10 }, (_, timestep) => (
                    <TabsContent key={timestep} value={timestep.toString()} className="space-y-4">
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                        {FEATURE_NAMES.map((feature, idx) => (
                          <div key={idx} className="space-y-1.5">
                            <Label htmlFor={`t${timestep}-f${idx}`} className="text-xs text-muted-foreground">
                              {feature}
                            </Label>
                            <Input
                              id={`t${timestep}-f${idx}`}
                              type="number"
                              step="0.0001"
                              value={sequences[timestep][idx]}
                              onChange={(e) => handleFeatureChange(timestep, idx, e.target.value)}
                              className="h-9 text-sm bg-background/50"
                            />
                          </div>
                        ))}
                      </div>
                    </TabsContent>
                  ))}
                </Tabs>

                <div className="mt-6 pt-6 border-t border-border/50">
                  <Button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="w-full h-12 text-lg font-semibold bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-opacity"
                  >
                    {isAnalyzing ? (
                      <>
                        <Activity className="w-5 h-5 mr-2 animate-spin" />
                        Analyzing Network Traffic...
                      </>
                    ) : (
                      <>
                        <Shield className="w-5 h-5 mr-2" />
                        Detect Attack
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right Panel - Results */}
          <div className="space-y-6">
            {/* Prediction Result */}
            {prediction && (
              <Card className={`border-2 ${ATTACK_INFO[prediction.attack as keyof typeof ATTACK_INFO].color} bg-card/50 backdrop-blur-sm shadow-2xl animate-in fade-in slide-in-from-right duration-500`}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    {(() => {
                      const Icon = ATTACK_INFO[prediction.attack as keyof typeof ATTACK_INFO].icon;
                      return <Icon className="w-6 h-6" />;
                    })()}
                    Detection Result
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Attack Type</div>
                    <div className="text-3xl font-bold uppercase tracking-wider">
                      {prediction.attack}
                    </div>
                  </div>
                  
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Confidence</div>
                    <div className="text-4xl font-bold">
                      {(prediction.confidence * 100).toFixed(2)}%
                    </div>
                    <div className="mt-2 h-2 bg-background/50 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-1000"
                        style={{ width: `${prediction.confidence * 100}%` }}
                      />
                    </div>
                  </div>

                  <Alert className="border-border/50 bg-background/50">
                    <AlertDescription className="text-xs">
                      {ATTACK_INFO[prediction.attack as keyof typeof ATTACK_INFO].description}
                    </AlertDescription>
                  </Alert>

                  <div className="space-y-2">
                    <div className="text-sm font-semibold text-muted-foreground">All Probabilities</div>
                    {Object.entries(prediction.probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([attack, prob]) => (
                        <div key={attack} className="flex items-center justify-between text-sm">
                          <span className="capitalize">{attack}</span>
                          <span className="font-mono">{(prob * 100).toFixed(2)}%</span>
                        </div>
                      ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Model Info */}
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-lg">Model Information</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Architecture</span>
                  <span className="font-semibold">Efficient ST-GNN</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Parameters</span>
                  <span className="font-semibold">39,749</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Test Accuracy</span>
                  <span className="font-semibold text-green-400">99.09%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">F1-Macro</span>
                  <span className="font-semibold text-green-400">98.04%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Inference Time</span>
                  <span className="font-semibold">&lt;1ms</span>
                </div>
              </CardContent>
            </Card>

            {/* Attack Types Legend */}
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-lg">Attack Types</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {Object.entries(ATTACK_INFO).map(([attack, info]) => {
                  const Icon = info.icon;
                  return (
                    <div key={attack} className="flex items-start gap-3">
                      <div className={`p-2 rounded-lg ${info.color}`}>
                        <Icon className="w-4 h-4" />
                      </div>
                      <div>
                        <div className="font-semibold capitalize text-sm">{attack}</div>
                        <div className="text-xs text-muted-foreground">{info.description}</div>
                      </div>
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      {/* Custom scrollbar styles */}
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: oklch(0.12 0.03 240);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: oklch(0.65 0.25 240);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: oklch(0.75 0.25 240);
        }
      `}</style>
    </div>
  );
}

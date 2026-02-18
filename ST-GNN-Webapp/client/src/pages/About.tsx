import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Satellite, Database, Brain, Target, Radio, Zap, AlertTriangle, ArrowLeft } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import { Link } from "wouter";

export default function About() {
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
                Orbit-Guard
              </h1>
              <p className="text-sm text-muted-foreground">Dataset Generation & Model Architecture</p>
            </div>
          </div>
          <Link href="/">
            <Button variant="outline" size="sm" className="gap-2">
              <ArrowLeft className="w-4 h-4" />
              Back to Detector
            </Button>
          </Link>
          </div>
        </div>
      </header>

      <main className="relative container py-8 space-y-8">
        {/* Overview */}
        <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-2xl flex items-center gap-2">
              <Brain className="w-6 h-6 text-primary" />
              Project Overview
            </CardTitle>
            <CardDescription>
              A novel Spatio-Temporal Graph Neural Network for detecting cyber attacks in LEO satellite networks
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-foreground/90 leading-relaxed">
              This project implements an advanced machine learning system for real-time detection of cyber attacks in Low Earth Orbit (LEO) satellite constellations. 
              Using a custom Spatio-Temporal Graph Neural Network (ST-GNN), the system achieves <strong className="text-primary">99.09% accuracy</strong> in 
              identifying Blackhole, DDoS, and Sinkhole attacks while maintaining excellent performance across all attack classes.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-primary">99.09%</div>
                <div className="text-sm text-muted-foreground">Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-primary">98.04%</div>
                <div className="text-sm text-muted-foreground">F1-Macro</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-primary">39.7K</div>
                <div className="text-sm text-muted-foreground">Parameters</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-primary">&lt;1ms</div>
                <div className="text-sm text-muted-foreground">Inference</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Dataset Generation */}
        <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-2xl flex items-center gap-2">
              <Database className="w-6 h-6 text-primary" />
              Dataset Generation Methodology
            </CardTitle>
            <CardDescription>
              NS-3 simulation of LEO satellite network with injected cyber attacks
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Simulation Environment */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Satellite className="w-5 h-5 text-accent" />
                Simulation Environment
              </h3>
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Simulator:</span>
                    <span className="font-semibold">NS-3 (Network Simulator 3)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Satellites:</span>
                    <span className="font-semibold">22 nodes (ID 120-141)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">ISL Links:</span>
                    <span className="font-semibold">453 edges</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Network Density:</span>
                    <span className="font-semibold">95.45%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Total Records:</span>
                    <span className="font-semibold">84,120 flows</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Sequences:</span>
                    <span className="font-semibold">16,767 (10 timesteps)</span>
                  </div>
                </div>
              </div>
            </div>

            <Separator />

            {/* Pipeline Steps */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Data Generation Pipeline</h3>
              <div className="space-y-3">
                <div className="flex gap-3 items-start">
                  <Badge className="mt-1">Step 1</Badge>
                  <div>
                    <div className="font-semibold text-sm">Traffic Flow Generation</div>
                    <div className="text-sm text-muted-foreground">
                      UDP flows between satellites with metrics: bytes sent/received, loss ratio, throughput
                    </div>
                  </div>
                </div>
                <div className="flex gap-3 items-start">
                  <Badge className="mt-1">Step 2</Badge>
                  <div>
                    <div className="font-semibold text-sm">ISL State Tracking</div>
                    <div className="text-sm text-muted-foreground">
                      Monitor link utilization, congestion, and hot link detection across all ISLs
                    </div>
                  </div>
                </div>
                <div className="flex gap-3 items-start">
                  <Badge className="mt-1">Step 3</Badge>
                  <div>
                    <div className="font-semibold text-sm">Routing Path Analysis</div>
                    <div className="text-sm text-muted-foreground">
                      Track hop count, path changes, and compromised satellite presence using trace_effective_fstate.py
                    </div>
                  </div>
                </div>
                <div className="flex gap-3 items-start">
                  <Badge className="mt-1">Step 4</Badge>
                  <div>
                    <div className="font-semibold text-sm">Cross-Layer Merging</div>
                    <div className="text-sm text-muted-foreground">
                      Align flow, ISL, and routing data by time bins; compute 24 engineered features
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Attack Types */}
        <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-2xl flex items-center gap-2">
              <Target className="w-6 h-6 text-primary" />
              Simulated Attack Types
            </CardTitle>
            <CardDescription>
              Three cyber attack scenarios injected into the satellite network
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Blackhole */}
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="p-3 rounded-lg bg-red-500/20 border border-red-500/50">
                  <AlertTriangle className="w-6 h-6 text-red-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold">Blackhole Attack</h3>
                  <Badge variant="outline" className="text-xs">7,892 samples (9.38%)</Badge>
                </div>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Compromised satellite advertises shortest paths to attract traffic, then drops all received packets without forwarding. 
                Creates routing instability as network attempts to route around the blackhole.
              </p>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-muted-foreground">Path Length:</span>
                  <span className="ml-2 font-semibold">6.98 hops (high)</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Throughput:</span>
                  <span className="ml-2 font-semibold">0.44 Mbps (low)</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Loss Ratio:</span>
                  <span className="ml-2 font-semibold">97.09% (very high)</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Path Changes:</span>
                  <span className="ml-2 font-semibold">55.02%</span>
                </div>
              </div>
            </div>

            <Separator />

            {/* DDoS */}
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="p-3 rounded-lg bg-orange-500/20 border border-orange-500/50">
                  <Zap className="w-6 h-6 text-orange-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold">DDoS Attack</h3>
                  <Badge variant="outline" className="text-xs">8,615 samples (10.24%)</Badge>
                </div>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Multiple compromised satellites flood target with high-rate traffic, overwhelming processing capacity and causing 
                severe congestion on links leading to the target satellite.
              </p>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-muted-foreground">Path Length:</span>
                  <span className="ml-2 font-semibold">4.09 hops (normal)</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Throughput:</span>
                  <span className="ml-2 font-semibold">1.61 Mbps (highest)</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Loss Ratio:</span>
                  <span className="ml-2 font-semibold">95.89% (high)</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Path Changes:</span>
                  <span className="ml-2 font-semibold">54.28%</span>
                </div>
              </div>
            </div>

            <Separator />

            {/* Sinkhole */}
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="p-3 rounded-lg bg-purple-500/20 border border-purple-500/50">
                  <Radio className="w-6 h-6 text-purple-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold">Sinkhole Attack</h3>
                  <Badge variant="outline" className="text-xs">8,343 samples (9.92%)</Badge>
                </div>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Compromised satellite advertises best routes to all destinations, attracting traffic from many sources. 
                Can eavesdrop, selectively forward, drop, or modify packets passing through.
              </p>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-muted-foreground">Path Length:</span>
                  <span className="ml-2 font-semibold">8.92 hops (highest)</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Throughput:</span>
                  <span className="ml-2 font-semibold">0.37 Mbps (lowest)</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Loss Ratio:</span>
                  <span className="ml-2 font-semibold">97.56% (highest)</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Bad Satellite:</span>
                  <span className="ml-2 font-semibold">94.82% presence</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Model Architecture */}
        <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-2xl flex items-center gap-2">
              <Brain className="w-6 h-6 text-primary" />
              ST-GNN Architecture
            </CardTitle>
            <CardDescription>
              Efficient Spatio-Temporal Graph Neural Network with attention mechanisms
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3 text-sm">
              <div className="p-4 bg-background/50 rounded-lg border border-border/50">
                <div className="font-mono text-xs space-y-1 text-muted-foreground">
                  <div>Input (batch, 10 timesteps, 24 features)</div>
                  <div className="ml-4">↓</div>
                  <div>Feature Embedding (64-dim) + BatchNorm + ReLU + Dropout(0.3)</div>
                  <div className="ml-4">↓</div>
                  <div>Temporal Processing</div>
                  <div className="ml-4">├─ Bidirectional GRU (64 → 32×2)</div>
                  <div className="ml-4">└─ Attention Mechanism (focus on critical timesteps)</div>
                  <div className="ml-4">↓</div>
                  <div>Spatial Processing</div>
                  <div className="ml-4">└─ Graph Convolution (2 layers, 64-dim)</div>
                  <div className="ml-4">↓</div>
                  <div>Feature Fusion (128-dim)</div>
                  <div className="ml-4">↓</div>
                  <div>Classification Head (128 → 64 → 4 classes)</div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 pt-2">
                <div>
                  <div className="text-muted-foreground mb-1">Training Time</div>
                  <div className="font-semibold">~5 minutes (30 epochs on CPU)</div>
                </div>
                <div>
                  <div className="text-muted-foreground mb-1">Class Balancing</div>
                  <div className="font-semibold">Focal Loss (γ=2.0) + Weighted Sampling</div>
                </div>
                <div>
                  <div className="text-muted-foreground mb-1">Optimizer</div>
                  <div className="font-semibold">AdamW (lr=0.001, wd=1e-4)</div>
                </div>
                <div>
                  <div className="text-muted-foreground mb-1">Regularization</div>
                  <div className="font-semibold">Dropout, BatchNorm, Gradient Clipping</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Features */}
        <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-2xl">Feature Engineering</CardTitle>
            <CardDescription>
              24 domain-specific features capturing attack signatures
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6 text-sm">
              <div className="space-y-3">
                <h4 className="font-semibold text-primary">Flow-Level (3)</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• loss_ratio - Packet loss ratio</li>
                  <li>• throughput_Mbps - Throughput in Mbps</li>
                  <li>• window_s - Time window duration</li>
                </ul>

                <h4 className="font-semibold text-primary mt-4">Routing (3)</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• path_len - Number of hops</li>
                  <li>• includes_s_bad - Bad satellite in path</li>
                  <li>• path_changed - Route changed</li>
                </ul>

                <h4 className="font-semibold text-primary mt-4">ISL Metrics (5)</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• isl_util_mean_path - Mean utilization</li>
                  <li>• isl_util_max_path - Max utilization</li>
                  <li>• isl_util_std_path - Std utilization</li>
                  <li>• isl_delta_mean_path - Delta utilization</li>
                  <li>• path_congestion_score - Congestion metric</li>
                </ul>
              </div>

              <div className="space-y-3">
                <h4 className="font-semibold text-primary">Engineered Indicators (13)</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• throughput_anomaly - Z-score deviation</li>
                  <li>• loss_spike - High loss indicator</li>
                  <li>• zero_throughput - Zero throughput flag</li>
                  <li>• path_efficiency - Inverse path length</li>
                  <li>• path_anomaly_score - Path × loss</li>
                  <li>• routing_instability - Path change flag</li>
                  <li>• isl_congestion_level - Categorical (0-3)</li>
                  <li>• isl_variance_ratio - Variance ratio</li>
                  <li>• bad_satellite_indicator - Bad sat flag</li>
                  <li>• high_path_len_indicator - Long path flag</li>
                  <li>• low_throughput_indicator - Low throughput flag</li>
                  <li>• congestion_loss_interaction - Cross-layer</li>
                  <li>• path_util_interaction - Cross-layer</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}

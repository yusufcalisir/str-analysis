"use client";

import { useMemo, useState, useEffect, useRef, useCallback } from "react";
import { MapContainer, TileLayer, Circle, CircleMarker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface GeoProbability {
    region: string;
    lat: number;
    lng: number;
    probability: number;
    color: string;
    initial_radius_km?: number;
    final_radius_km?: number;
}

type ScanPhase = "idle" | "scanning" | "calculating" | "locked";

// ═══════════════════════════════════════════════════════════════════════════════
// HEATMAP LAYER (leaflet.heat integration)
// ═══════════════════════════════════════════════════════════════════════════════

function HeatmapLayer({ data }: { data: GeoProbability[] }) {
    const map = useMap();
    const isMounted = useRef(true);
    const layerRef = useRef<L.Layer | null>(null);

    useEffect(() => {
        isMounted.current = true;
        return () => {
            isMounted.current = false;
        };
    }, []);

    useEffect(() => {
        if (!data || data.length === 0) return;

        import("leaflet.heat").then(() => {
            if (!isMounted.current || !map) return;

            // Remove existing layer if it exists (e.g. data update)
            if (layerRef.current) {
                map.removeLayer(layerRef.current);
                layerRef.current = null;
            }

            const points: [number, number, number][] = data.map((d) => [
                d.lat,
                d.lng,
                d.probability,
            ]);

            // @ts-ignore — leaflet.heat extends L globally
            const layer = L.heatLayer(points, {
                radius: 45,
                blur: 30,
                maxZoom: 6,
                max: 1.0,
                minOpacity: 0.15,
                gradient: {
                    0.0: "rgba(0,0,0,0)",
                    0.2: "#1a1a4e", // Deep Navy
                    0.4: "#3B82F6", // Blue
                    0.6: "#22C55E", // Green
                    0.8: "#F59E0B", // Amber
                    1.0: "#EF4444", // Red
                },
            });

            layer.addTo(map);
            layerRef.current = layer;
        });

        return () => {
            if (layerRef.current) {
                map.removeLayer(layerRef.current);
                layerRef.current = null;
            }
        };
    }, [data, map]);

    return null;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCAN CONTROLLER — drives flyTo zoom + phase transitions
// ═══════════════════════════════════════════════════════════════════════════════

function ScanController({
    target,
    phase,
    onPhaseChange,
}: {
    target: GeoProbability;
    phase: ScanPhase;
    onPhaseChange: (phase: ScanPhase) => void;
}) {
    const map = useMap();
    const startedRef = useRef(false);

    useEffect(() => {
        if (phase !== "idle" || startedRef.current) return;
        startedRef.current = true;

        // Phase 1: Scanning Global Databases (0–1s)
        onPhaseChange("scanning");
        map.flyTo([target.lat, target.lng], 4, { duration: 1.2 });

        // Phase 2: Calculating Population Covariance (1–2s)
        const t1 = setTimeout(() => {
            onPhaseChange("calculating");
            map.flyTo([target.lat, target.lng], 7, { duration: 1.0 });
        }, 1200);

        // Phase 3: 95% Confidence Zone Locked (2–3s)
        const t2 = setTimeout(() => {
            onPhaseChange("locked");
            const finalZoom = target.final_radius_km && target.final_radius_km < 200 ? 10 : 8;
            map.flyTo([target.lat, target.lng], finalZoom, { duration: 0.8 });
        }, 2600);

        return () => {
            clearTimeout(t1);
            clearTimeout(t2);
        };
    }, [phase, target, map, onPhaseChange]);

    return null;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIDENCE RING — animated radius shrink from initial → final
// ═══════════════════════════════════════════════════════════════════════════════

function ConfidenceRing({
    region,
    phase,
}: {
    region: GeoProbability;
    phase: ScanPhase;
}) {
    const initialR = (region.initial_radius_km || 2000) * 1000; // km → m
    const finalR = (region.final_radius_km || 200) * 1000;
    const [currentRadius, setCurrentRadius] = useState(initialR);
    const animRef = useRef<number | null>(null);
    const startTimeRef = useRef<number | null>(null);

    const ANIMATION_DURATION = 2600; // ms (matches scan controller)

    useEffect(() => {
        if (phase === "idle") {
            setCurrentRadius(initialR);
            return;
        }

        if (phase === "scanning" && !startTimeRef.current) {
            startTimeRef.current = performance.now();
        }

        if (phase !== "locked" && startTimeRef.current) {
            const animate = (now: number) => {
                const elapsed = now - (startTimeRef.current || now);
                const progress = Math.min(elapsed / ANIMATION_DURATION, 1);

                // Ease-out cubic for smooth deceleration
                const eased = 1 - Math.pow(1 - progress, 3);
                const r = initialR + (finalR - initialR) * eased;
                setCurrentRadius(Math.max(r, finalR));

                if (progress < 1) {
                    animRef.current = requestAnimationFrame(animate);
                }
            };

            animRef.current = requestAnimationFrame(animate);
        }

        if (phase === "locked") {
            setCurrentRadius(finalR);
            startTimeRef.current = null;
        }

        return () => {
            if (animRef.current) cancelAnimationFrame(animRef.current);
        };
    }, [phase, initialR, finalR]);

    const isLocked = phase === "locked";
    const isActive = phase === "scanning" || phase === "calculating";

    return (
        <>
            {/* Confidence area circle */}
            <Circle
                center={[region.lat, region.lng]}
                radius={currentRadius}
                pathOptions={{
                    fillColor: "#22C55E",
                    fillOpacity: isLocked ? 0.12 : 0.06,
                    color: "#22C55E",
                    weight: isActive ? 2 : 1.5,
                    opacity: isActive ? 0.7 : 0.5,
                    dashArray: isActive ? "8 4" : undefined,
                    className: isActive ? "confidence-ring-pulse" : "",
                }}
            />

            {/* Centroid marker — blinks only when locked */}
            <CircleMarker
                center={[region.lat, region.lng]}
                radius={isLocked ? 6 : 4}
                pathOptions={{
                    fillColor: region.color,
                    fillOpacity: 1,
                    color: "#fafafa",
                    weight: isLocked ? 2 : 1,
                    opacity: 1,
                    className: isLocked ? "centroid-locked-blink" : "",
                }}
            >
                <Popup>
                    <div
                        style={{
                            fontFamily: "'JetBrains Mono', monospace",
                            fontSize: "11px",
                            color: "#fafafa",
                            background: "#111113",
                            padding: "12px 16px",
                            borderRadius: "6px",
                            border: `1px solid ${region.color}40`,
                            minWidth: "200px",
                        }}
                    >
                        <div
                            style={{
                                fontSize: "8px",
                                letterSpacing: "0.15em",
                                color: "#71717A",
                                textTransform: "uppercase",
                                marginBottom: "6px",
                            }}
                        >
                            95% CONFIDENCE ZONE
                        </div>
                        <div
                            style={{
                                fontSize: "13px",
                                fontWeight: "700",
                                color: region.color,
                                marginBottom: "4px",
                            }}
                        >
                            {region.region}
                        </div>
                        <div
                            style={{
                                fontSize: "18px",
                                fontWeight: "800",
                                color: "#fafafa",
                                marginBottom: "4px",
                            }}
                        >
                            {(region.probability * 100).toFixed(1)}%
                        </div>
                        <div style={{ fontSize: "9px", color: "#52525B", marginBottom: "2px" }}>
                            Radius: {(currentRadius / 1000).toFixed(0)} km
                        </div>
                        <div style={{ fontSize: "9px", color: "#52525B" }}>
                            {region.lat.toFixed(2)}°N, {region.lng.toFixed(2)}°E
                        </div>
                    </div>
                </Popup>
            </CircleMarker>
        </>
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECONDARY REGION MARKERS (ranks 2–3, static)
// ═══════════════════════════════════════════════════════════════════════════════

function SecondaryMarkers({ data }: { data: GeoProbability[] }) {
    const secondary = useMemo(() => data.slice(1, 3), [data]);

    return (
        <>
            {secondary.map((region, idx) => (
                <CircleMarker
                    key={region.region}
                    center={[region.lat, region.lng]}
                    radius={6}
                    pathOptions={{
                        fillColor: region.color,
                        fillOpacity: 0.5,
                        color: region.color,
                        weight: 1.5,
                        opacity: 0.6,
                    }}
                >
                    <Popup>
                        <div
                            style={{
                                fontFamily: "'JetBrains Mono', monospace",
                                fontSize: "11px",
                                color: "#fafafa",
                                background: "#111113",
                                padding: "12px 16px",
                                borderRadius: "6px",
                                border: `1px solid ${region.color}40`,
                                minWidth: "180px",
                            }}
                        >
                            <div style={{ fontSize: "8px", letterSpacing: "0.15em", color: "#71717A", textTransform: "uppercase", marginBottom: "6px" }}>
                                GEO-FORENSIC MATCH #{idx + 2}
                            </div>
                            <div style={{ fontSize: "13px", fontWeight: "700", color: region.color, marginBottom: "4px" }}>
                                {region.region}
                            </div>
                            <div style={{ fontSize: "18px", fontWeight: "800", color: "#fafafa" }}>
                                {(region.probability * 100).toFixed(1)}%
                            </div>
                        </div>
                    </Popup>
                </CircleMarker>
            ))}
        </>
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN MAP COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export default function ForensicMap({
    data,
    onScanPhaseChange,
}: {
    data: GeoProbability[];
    onScanPhaseChange?: (phase: ScanPhase) => void;
}) {
    const [phase, setPhase] = useState<ScanPhase>("idle");

    const center = useMemo<[number, number]>(() => {
        if (data.length > 0) return [data[0].lat, data[0].lng];
        return [30.0, 30.0];
    }, [data]);

    const topRegion = data[0] || null;

    const handlePhaseChange = useCallback(
        (newPhase: ScanPhase) => {
            setPhase(newPhase);
            onScanPhaseChange?.(newPhase);
        },
        [onScanPhaseChange]
    );

    return (
        <MapContainer
            center={center}
            zoom={2}
            scrollWheelZoom={true}
            zoomControl={false}
            attributionControl={false}
            style={{
                width: "100%",
                height: "100%",
                borderRadius: "0",
                background: "#0A0A0B",
            }}
        >
            {/* CartoDB Dark Matter */}
            <TileLayer
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                maxZoom={19}
            />

            {/* Heatmap overlay */}
            <HeatmapLayer data={data} />

            {/* Scan controller — drives flyTo + phase machine */}
            {topRegion && (
                <ScanController
                    target={topRegion}
                    phase={phase}
                    onPhaseChange={handlePhaseChange}
                />
            )}

            {/* Primary region: animated confidence ring */}
            {topRegion && (
                <ConfidenceRing region={topRegion} phase={phase} />
            )}

            {/* Secondary markers (ranks 2–3) */}
            <SecondaryMarkers data={data} />
        </MapContainer>
    );
}

export type { ScanPhase, GeoProbability };

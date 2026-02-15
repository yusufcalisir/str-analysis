"use client";

import { useMemo, useState, useEffect, useRef, useCallback } from "react";
import { MapContainer, TileLayer, Circle, CircleMarker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface GeoProbability {
    region: string;
    lat: number;
    lng: number;
    probability: number;
    color: string;
    initial_radius_km?: number;
    final_radius_km?: number;
}

export type ScanPhase = "idle" | "scanning" | "calculating" | "locked";

// ═══════════════════════════════════════════════════════════════════════════════
// HEATMAP LAYER
// ═══════════════════════════════════════════════════════════════════════════════

function HeatmapLayer({ data }: { data: GeoProbability[] }) {
    const map = useMap();
    const isMounted = useRef(true);
    const layerRef = useRef<L.Layer | null>(null);

    useEffect(() => {
        isMounted.current = true;
        return () => { isMounted.current = false; };
    }, []);

    useEffect(() => {
        if (!data || data.length === 0) return;

        import("leaflet.heat").then(() => {
            if (!isMounted.current || !map) return;

            if (layerRef.current) {
                map.removeLayer(layerRef.current);
                layerRef.current = null;
            }

            const points: [number, number, number][] = data.map((d) => [
                d.lat,
                d.lng,
                d.probability,
            ]);

            // @ts-ignore
            const layer = L.heatLayer(points, {
                radius: 45,
                blur: 30,
                maxZoom: 6,
                max: 1.0,
                minOpacity: 0.15,
                gradient: {
                    0.0: "rgba(0,0,0,0)",
                    0.2: "#1a1a4e",
                    0.4: "#3B82F6",
                    0.6: "#22C55E",
                    0.8: "#F59E0B",
                    1.0: "#EF4444",
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
// SCAN CONTROLLER
// ═══════════════════════════════════════════════════════════════════════════════

function ScanController({
    target,
    onPhaseChange,
}: {
    target: GeoProbability;
    onPhaseChange: (phase: ScanPhase) => void;
}) {
    const map = useMap();
    const currentTargetRef = useRef<string | null>(null);

    useEffect(() => {
        if (!target || target.region === currentTargetRef.current) return;
        currentTargetRef.current = target.region;

        onPhaseChange("scanning");

        const t1 = setTimeout(() => {
            map.flyTo([target.lat, target.lng], 4, { duration: 2.5 });
            onPhaseChange("calculating");
        }, 1000);

        const t2 = setTimeout(() => {
            map.flyTo([target.lat, target.lng], 5, { duration: 1.5 });
            onPhaseChange("locked");
        }, 4000);

        return () => {
            clearTimeout(t1);
            clearTimeout(t2);
        };
    }, [target, map, onPhaseChange]);

    return null;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIDENCE RING
// ═══════════════════════════════════════════════════════════════════════════════

function ConfidenceRing({
    region,
    phase,
    onHover,
}: {
    region: GeoProbability;
    phase: ScanPhase;
    onHover?: (region: string | null) => void;
}) {
    const initialR = (region.initial_radius_km || 2000) * 1000;
    const finalR = (region.final_radius_km || 200) * 1000;
    const [currentRadius, setCurrentRadius] = useState(initialR);
    const animRef = useRef<number | null>(null);
    const startTimeRef = useRef<number | null>(null);

    const ANIMATION_DURATION = 3000;

    useEffect(() => {
        if (phase === "idle") {
            setCurrentRadius(initialR);
            startTimeRef.current = null;
            return;
        }

        if (phase === "scanning" && !startTimeRef.current) {
            startTimeRef.current = performance.now();
        }

        if (phase !== "locked" && startTimeRef.current) {
            const animate = (now: number) => {
                const elapsed = now - (startTimeRef.current || now);
                const progress = Math.min(elapsed / ANIMATION_DURATION, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                const r = initialR + (finalR - initialR) * eased;
                setCurrentRadius(Math.max(r, finalR));
                if (progress < 1) animRef.current = requestAnimationFrame(animate);
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
            <Circle
                center={[region.lat, region.lng]}
                radius={currentRadius}
                eventHandlers={{
                    mouseover: () => onHover?.(region.region),
                    mouseout: () => onHover?.(null),
                    click: () => onHover?.(region.region),
                }}
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
            />
        </>
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN MAP COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export default function ForensicMap({
    data,
    kinshipMatches,
    onScanPhaseChange,
    onRegionHover,
}: {
    data: GeoProbability[];
    kinshipMatches?: any[];
    onScanPhaseChange?: (phase: ScanPhase) => void;
    onRegionHover?: (region: string | null) => void;
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
            <TileLayer
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                maxZoom={19}
            />

            <HeatmapLayer data={data} />

            {topRegion && (
                <ScanController
                    target={topRegion}
                    onPhaseChange={handlePhaseChange}
                />
            )}

            {topRegion && (
                <ConfidenceRing
                    region={topRegion}
                    phase={phase}
                    onHover={onRegionHover}
                />
            )}

            {data.slice(1, 3).map((region) => (
                <CircleMarker
                    key={region.region}
                    center={[region.lat, region.lng]}
                    radius={6}
                    eventHandlers={{
                        mouseover: () => onRegionHover?.(region.region),
                        mouseout: () => onRegionHover?.(null),
                        click: () => onRegionHover?.(region.region),
                    }}
                    pathOptions={{
                        fillColor: region.color,
                        fillOpacity: 0.5,
                        color: region.color,
                        weight: 1.5,
                        opacity: 0.6,
                    }}
                />
            ))}
        </MapContainer>
    );
}

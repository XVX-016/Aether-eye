"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

type AirflowLinesProps = {
    count?: number;
};

export const Particles = ({ count = 3000 }: AirflowLinesProps) => {
    const linesRef = useRef<THREE.LineSegments>(null);

    const { positions, speeds, lengths, vy, vz } = useMemo(() => {
        const pos = new Float32Array(count * 2 * 3);
        const spd = new Float32Array(count);
        const len = new Float32Array(count);
        const latVy = new Float32Array(count);
        const latVz = new Float32Array(count);

        for (let i = 0; i < count; i++) {
            const y = (Math.random() - 0.5) * 16;
            const z = (Math.random() - 0.5) * 30;
            const x = Math.random() * 84 - 42;
            const length = 2.2 + Math.random() * 3.6;

            const i6 = i * 6;
            pos[i6 + 0] = x;
            pos[i6 + 1] = y;
            pos[i6 + 2] = z;

            pos[i6 + 3] = x + length;
            pos[i6 + 4] = y;
            pos[i6 + 5] = z;

            spd[i] = 0.22 + Math.random() * 0.42;
            len[i] = length;
            latVy[i] = (Math.random() - 0.5) * 0.01;
            latVz[i] = (Math.random() - 0.5) * 0.01;
        }

        return { positions: pos, speeds: spd, lengths: len, vy: latVy, vz: latVz };
    }, [count]);

    useFrame(() => {
        if (!linesRef.current) return;

        const attr = linesRef.current.geometry.getAttribute("position");
        const arr = attr.array as Float32Array;

        for (let i = 0; i < count; i++) {
            const i6 = i * 6;
            const vel = speeds[i];

            // Base flow is right -> left.
            let vx = -vel;
            let curVy = vy[i];
            let curVz = vz[i];

            const headX = arr[i6 + 0];
            const headY = arr[i6 + 1];
            const headZ = arr[i6 + 2];

            // Aircraft envelope around visible fuselage.
            const inPlaneZone =
                headX > -4.5 &&
                headX < 8.2 &&
                Math.abs(headY) < 3.1 &&
                Math.abs(headZ) < 6.4;

            if (inPlaneZone) {
                const fuselageDist = Math.sqrt((headY * 0.9) ** 2 + (headZ * 0.55) ** 2);
                const repel = Math.max(0, 1.5 - fuselageDist);

                // Repulsion away from body gives visible deflection arc.
                curVy += (headY >= 0 ? 1 : -1) * repel * 0.03;
                curVz += (headZ >= 0 ? 1 : -1) * repel * 0.05;

                // Slight drag near body to emphasize interaction.
                vx *= 0.93;
            }

            // Wing-tip style vortex zones for curl effect.
            const tipL = { x: 1.8, y: 0.0, z: -4.7 };
            const tipR = { x: 1.8, y: 0.0, z: 4.7 };

            const applyVortex = (tip: { x: number; y: number; z: number }, sign: number) => {
                const dx = headX - tip.x;
                const dy = headY - tip.y;
                const dz = headZ - tip.z;
                const d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < 14.0) {
                    const s = (14.0 - d2) / 14.0;
                    // Swirl in yz plane to look like air peeling off wing.
                    curVy += (-dz * sign) * 0.006 * s;
                    curVz += (dy * sign) * 0.006 * s;
                }
            };

            applyVortex(tipL, -1);
            applyVortex(tipR, 1);

            // Damping to keep motion stable.
            curVy *= 0.985;
            curVz *= 0.985;

            vy[i] = curVy;
            vz[i] = curVz;

            arr[i6 + 0] += vx;
            arr[i6 + 1] += curVy;
            arr[i6 + 2] += curVz;

            // Tail follows velocity direction so streaks can visibly bend.
            const mag = Math.sqrt(vx * vx + curVy * curVy + curVz * curVz) || 1;
            const dirX = vx / mag;
            const dirY = curVy / mag;
            const dirZ = curVz / mag;
            const length = lengths[i];

            arr[i6 + 3] = arr[i6 + 0] - dirX * length;
            arr[i6 + 4] = arr[i6 + 1] - dirY * length;
            arr[i6 + 5] = arr[i6 + 2] - dirZ * length;

            // Re-seed when streak exits left bound or drifts too far out.
            if (
                arr[i6 + 0] < -44 ||
                Math.abs(arr[i6 + 1]) > 13.5 ||
                Math.abs(arr[i6 + 2]) > 22.0
            ) {
                const y = (Math.random() - 0.5) * 16;
                const z = (Math.random() - 0.5) * 30;
                const length = 2.2 + Math.random() * 3.6;
                const x = 44 + Math.random() * 6;

                arr[i6 + 0] = x;
                arr[i6 + 1] = y;
                arr[i6 + 2] = z;

                arr[i6 + 3] = x + length;
                arr[i6 + 4] = y;
                arr[i6 + 5] = z;

                lengths[i] = length;
                vy[i] = (Math.random() - 0.5) * 0.01;
                vz[i] = (Math.random() - 0.5) * 0.01;
            }
        }

        attr.needsUpdate = true;
    });

    return (
        <lineSegments ref={linesRef} frustumCulled={false}>
            <bufferGeometry>
                <bufferAttribute
                    attach="attributes-position"
                    args={[positions, 3]}
                    count={positions.length / 3}
                    itemSize={3}
                />
            </bufferGeometry>
            <lineBasicMaterial
                color="#cbd5e1"
                transparent
                opacity={0.34}
                blending={THREE.AdditiveBlending}
                depthWrite={false}
            />
        </lineSegments>
    );
};

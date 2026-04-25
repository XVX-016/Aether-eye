"use client";

import { useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import type { Group } from "three";

export const HomepageTakeoff = () => {
    const planeRef = useRef<Group>(null);
    const { size } = useThree();

    const isMobile = size.width < 768;
    const basePosition: [number, number, number] = isMobile ? [1.2, -0.05, 0] : [1.9, 0.05, 0];

    const heroJet = useMemo(
        () => (
            <group scale={isMobile ? 1.75 : 2.55}>
                <mesh castShadow receiveShadow rotation={[0, 0, -Math.PI / 2]}>
                    <cylinderGeometry args={[0.24, 0.34, 3.8, 24]} />
                    <meshStandardMaterial color="#dbeafe" emissive="#082f49" emissiveIntensity={0.22} metalness={0.75} roughness={0.22} />
                </mesh>
                <mesh castShadow receiveShadow position={[2.05, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
                    <coneGeometry args={[0.34, 1.15, 24]} />
                    <meshStandardMaterial color="#f8fafc" metalness={0.82} roughness={0.18} />
                </mesh>
                <mesh castShadow receiveShadow position={[-1.55, 0.42, 0]} rotation={[0, 0, -Math.PI / 2]}>
                    <boxGeometry args={[0.88, 0.08, 1.12]} />
                    <meshStandardMaterial color="#7dd3fc" emissive="#0c4a6e" emissiveIntensity={0.45} metalness={0.4} roughness={0.28} />
                </mesh>
                <mesh castShadow receiveShadow position={[0.2, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
                    <boxGeometry args={[1.75, 0.12, 4.9]} />
                    <meshStandardMaterial color="#cbd5e1" metalness={0.55} roughness={0.32} />
                </mesh>
                <mesh castShadow receiveShadow position={[-0.85, 0.16, 0]} rotation={[0, 0, -Math.PI / 2]}>
                    <boxGeometry args={[0.92, 0.08, 2.3]} />
                    <meshStandardMaterial color="#94a3b8" metalness={0.48} roughness={0.35} />
                </mesh>
                <mesh castShadow receiveShadow position={[0.35, 0.1, 1.38]} rotation={[0.18, 0, -Math.PI / 2]}>
                    <coneGeometry args={[0.09, 0.95, 10]} />
                    <meshStandardMaterial color="#38bdf8" emissive="#38bdf8" emissiveIntensity={0.8} metalness={0.55} roughness={0.18} />
                </mesh>
                <mesh castShadow receiveShadow position={[0.35, 0.1, -1.38]} rotation={[-0.18, 0, -Math.PI / 2]}>
                    <coneGeometry args={[0.09, 0.95, 10]} />
                    <meshStandardMaterial color="#38bdf8" emissive="#38bdf8" emissiveIntensity={0.8} metalness={0.55} roughness={0.18} />
                </mesh>
                <mesh position={[2.35, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
                    <coneGeometry args={[0.12, 0.7, 12]} />
                    <meshBasicMaterial color="#ffffff" transparent opacity={0.22} />
                </mesh>
            </group>
        ),
        [isMobile],
    );

    useFrame(({ clock }) => {
        if (!planeRef.current) return;

        const t = clock.getElapsedTime();
        planeRef.current.rotation.y = -0.58 + Math.sin(t * 0.2) * 0.08;
        planeRef.current.rotation.z = -0.07 + Math.sin(t * 0.42) * 0.025;
        planeRef.current.rotation.x = 0.08 + Math.sin(t * 0.28) * 0.012;
        planeRef.current.position.y = basePosition[1] + Math.sin(t * 0.5) * 0.04;
        planeRef.current.position.x = basePosition[0] + Math.sin(t * 0.18) * 0.08;
        planeRef.current.position.z = basePosition[2] + Math.sin(t * 0.22) * 0.04;
    });

    return (
        <group>
            <spotLight position={[7, 7, 9]} angle={0.55} penumbra={0.9} intensity={5.5} castShadow={false} />
            <pointLight position={[-6, -2, -4]} intensity={0.8} color="#38bdf8" />
            <pointLight position={[4, 2, 7]} intensity={1.7} color="#ffffff" />
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[1.2, -1.55, 0]}>
                <planeGeometry args={[16, 10]} />
                <meshStandardMaterial color="#08111d" metalness={0.15} roughness={0.92} />
            </mesh>

            <group
                ref={planeRef}
                rotation={[0.08, -0.58, -0.06]}
                position={basePosition}
                scale={1}
            >
                {heroJet}
            </group>
        </group>
    );
};

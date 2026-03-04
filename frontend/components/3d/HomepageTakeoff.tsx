"use client";

import { useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import type { Group } from "three";

export const HomepageTakeoff = () => {
    const planeRef = useRef<Group>(null);
    const { scene } = useGLTF("/models/fighterplane/scene.gltf");
    const { size } = useThree();

    const isMobile = size.width < 768;
    const modelScale = isMobile ? 1.55 : 2.65;

    useFrame(({ clock }) => {
        if (!planeRef.current) return;

        const t = clock.getElapsedTime();
        planeRef.current.rotation.y = -Math.PI / 2 + Math.sin(t * 0.24) * 0.015;
        planeRef.current.rotation.z = Math.sin(t * 0.42) * 0.012;
        planeRef.current.rotation.x = -0.04 + Math.sin(t * 0.3) * 0.01;
        planeRef.current.position.y = Math.sin(t * 0.6) * 0.035;
        planeRef.current.position.x = 0.03 + Math.sin(t * 0.2) * 0.04;
        planeRef.current.position.z = -0.12;
    });

    return (
        <group>
            <spotLight position={[5, 10, 10]} angle={0.5} penumbra={1} intensity={5} castShadow />
            <pointLight position={[-10, -10, -10]} intensity={2} color="#00e680" />

            <group ref={planeRef} rotation={[0, -Math.PI / 2, 0]} position={[0, 0, 0]}>
                <primitive
                    object={scene}
                    scale={isMobile ? 1.5 : 2.5}
                    position={[0, 0, 0]}
                    rotation={[0, Math.PI, 0]}
                />
            </group>
        </group>
    );
};

useGLTF.preload("/models/fighterplane/scene.gltf");

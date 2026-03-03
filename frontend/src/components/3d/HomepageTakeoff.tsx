import { useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import type { Group } from "three";

export const HomepageTakeoff = () => {
  const planeRef = useRef<Group>(null);
  const { scene } = useGLTF("/models/fighterplane/scene.gltf");
  const { size } = useThree();

  const isMobile = size.width < 768;
  const modelScale = isMobile ? 1.45 : 2.35;

  useFrame(({ clock }) => {
    if (!planeRef.current) return;
    const t = clock.getElapsedTime();
    planeRef.current.rotation.y = -Math.PI / 2 + Math.sin(t * 0.45) * 0.04;
    planeRef.current.position.y = Math.sin(t * 0.9) * 0.12;
  });

  return (
    <group>
      <spotLight
        position={[5, 10, 10]}
        angle={0.5}
        penumbra={1}
        intensity={4.5}
        castShadow
      />
      <pointLight position={[-10, -8, -10]} intensity={1.8} color="#00d18f" />

      <group ref={planeRef} rotation={[0, -Math.PI / 2, 0]} position={[0, 0, 0]}>
        <primitive
          object={scene}
          scale={modelScale}
          position={[0, 0, 0]}
          rotation={[0, Math.PI, 0]}
        />
      </group>
    </group>
  );
};

useGLTF.preload("/models/fighterplane/scene.gltf");

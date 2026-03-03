import { Suspense, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Environment } from "@react-three/drei";
import * as THREE from "three";
import { HomepageTakeoff } from "../3d/HomepageTakeoff";
import { HomepageNav } from "../home/HomepageNav";

const ResponsiveCamera = () => {
  const { size, camera } = useThree();
  const target = useRef(new THREE.Vector3(5, 2, 8));
  const settled = useRef(false);
  const lastMobile = useRef<boolean | null>(null);

  useFrame(() => {
    const isMobile = size.width < 768;

    if (lastMobile.current !== isMobile) {
      lastMobile.current = isMobile;
      settled.current = false;
      target.current.set(
        isMobile ? 3 : 5,
        isMobile ? 4 : 2,
        isMobile ? 20 : 8,
      );
    }

    if (settled.current) return;

    const dist = camera.position.distanceTo(target.current);
    if (dist > 0.01) {
      camera.position.lerp(target.current, 0.1);
      camera.lookAt(0, 0, 0);
    } else {
      camera.position.copy(target.current);
      camera.lookAt(0, 0, 0);
      camera.updateProjectionMatrix();
      settled.current = true;
    }
  });

  return null;
};

export const HeroSection = () => {
  return (
    <section className="hero-section">
      <HomepageNav />
      <div className="hero-canvas-wrap">
        <Canvas
          dpr={1}
          gl={{
            antialias: true,
            powerPreference: "high-performance",
            alpha: false,
          }}
          camera={{ position: [5, 2, 8], fov: 50, near: 0.1, far: 5000 }}
        >
          <color attach="background" args={["#020617"]} />
          <Suspense fallback={null}>
            <Environment preset="city" background={false} />
            <ambientLight intensity={0.4} />
            <directionalLight position={[5, 10, 5]} intensity={1.2} />
            <ResponsiveCamera />
            <HomepageTakeoff />
          </Suspense>
        </Canvas>
      </div>

      <div className="hero-overlay" />
      <div className="hero-fade" />
      <div className="hero-content">
        <p className="hero-kicker mono">AETHER EYE INTELLIGENCE SYSTEM</p>
        <h2 className="hero-title">Precision Flight</h2>
        <h3 className="hero-subtitle">Dynamics And Control</h3>
        <p className="hero-support mono">AEROSPACE INTELLIGENCE OPERATIONS SURFACE</p>
      </div>
    </section>
  );
};

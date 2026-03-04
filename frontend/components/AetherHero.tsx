"use client";

import { Suspense, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Environment } from "@react-three/drei";
import * as THREE from "three";
import { HomepageTakeoff } from "./3d/HomepageTakeoff";
import { Particles } from "./3d/Particles";

const ResponsiveCamera = () => {
  const { camera, size } = useThree();
  const target = useRef(new THREE.Vector3(5, 2, 8));

  useFrame(() => {
    const isMobile = size.width < 768;
    target.current.set(isMobile ? 3 : 5, isMobile ? 4 : 2, isMobile ? 20 : 8);
    camera.position.lerp(target.current, 0.1);
    camera.lookAt(0, 0, 0);
  });

  return null;
};

export const AetherHero = () => {
  return (
    <section className="hero-section">
      <div className="hero-canvas-wrap">
        <Canvas
          dpr={1}
          gl={{ antialias: true, powerPreference: "high-performance", alpha: false }}
          camera={{ position: [5, 2, 8], fov: 50, near: 0.1, far: 5000 }}
        >
          <color attach="background" args={["#020408"]} />
          <fog attach="fog" args={["#020408", 8, 45]} />
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 10, 5]} intensity={1.2} />
          <Environment preset="city" />
          <Suspense fallback={null}>
            <Particles count={4400} />
            <ResponsiveCamera />
            <HomepageTakeoff />
          </Suspense>
        </Canvas>
      </div>

      <div className="hero-streaks" aria-hidden="true">
        {Array.from({ length: 14 }).map((_, idx) => (
          <span key={idx} className={`hero-streak hero-streak-${(idx % 6) + 1}`} />
        ))}
      </div>

      <div className="hero-content">
        <p className="hero-kicker mono">AETHER EYE INTELLIGENCE SYSTEM</p>

        <div className="hero-typography">
          <h1 className="hero-main-title">
            <span className="hero-word-bold">Intelligence</span>
            <span className="hero-word-bold">Driven</span>
          </h1>
          <h2 className="hero-sub-title hero-outline-title">Detection &amp; Analysis</h2>
        </div>

        <p className="hero-support mono">AETHER EYE INTELLIGENCE OPERATIONS SURFACE</p>

        <div className="hero-actions">
          <button className="hero-cta glass">ENTER CONSOLE</button>
        </div>
      </div>

      <div className="hero-fade" />
    </section>
  );
};

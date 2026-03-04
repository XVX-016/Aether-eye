import "./globals.css";
import { CountryProvider } from "@/context/CountryContext";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <CountryProvider>{children}</CountryProvider>
      </body>
    </html>
  );
}

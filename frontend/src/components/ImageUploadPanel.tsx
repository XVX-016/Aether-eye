import React from "react";
import { ContentPanel } from "./ContentPanel";

type Props = {
  label: string;
  file: File | null;
  onChange: (file: File | null) => void;
  accept?: string;
  helpText?: string;
};

export const ImageUploadPanel: React.FC<Props> = ({
  label,
  file,
  onChange,
  accept = "image/*",
  helpText,
}) => {
  return (
    <ContentPanel title={label} subtitle={helpText}>
      <label className="file-input">
        <input
          type="file"
          accept={accept}
          onChange={(e) => onChange(e.target.files?.[0] ?? null)}
        />
        <span className="file-input-button">Choose image</span>
        <span className="file-input-name">{file ? file.name : "No file selected"}</span>
      </label>
    </ContentPanel>
  );
};


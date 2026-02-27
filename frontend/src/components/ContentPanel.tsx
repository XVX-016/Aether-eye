import React from "react";

type Props = {
  title: string;
  subtitle?: string;
  headerRight?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
};

export const ContentPanel: React.FC<Props> = ({
  title,
  subtitle,
  headerRight,
  className,
  children,
}) => {
  return (
    <section className={className ? `panel ${className}` : "panel"}>
      <header className="panel-header">
        <div className="panel-title-wrap">
          <div className="panel-title">{title}</div>
          {subtitle && <div className="panel-subtitle">{subtitle}</div>}
        </div>
        {headerRight && <div className="panel-actions">{headerRight}</div>}
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
};

import type { ReactNode } from "react";
import type { PageName } from "./Navbar";
import Navbar from "./Navbar";

interface PageContainerProps {
  activePage: PageName;
  children: ReactNode;
}

export default function PageContainer({
  activePage,
  children,
}: PageContainerProps) {
  return (
    <>
      <Navbar activePage={activePage} />
      <div className="px-12 pt-24 pb-6">{children}</div>
    </>
  );
}

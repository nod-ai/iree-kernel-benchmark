import React, { type JSX } from "react";
import { FaHome, FaHistory, FaPlus, FaGuitar } from "react-icons/fa";
import { twMerge } from "tailwind-merge";

export type PageName = "dashboard" | "history" | "new" | "tune";

interface NavbarProps {
  activePage: PageName;
}

const navItems: { name: PageName; label: string; icon: JSX.Element }[] = [
  { name: "dashboard", label: "Dashboard", icon: <FaHome /> },
  { name: "tune", label: "Tuning", icon: <FaGuitar /> },
  { name: "history", label: "History", icon: <FaHistory /> },
  { name: "new", label: "Add Kernels", icon: <FaPlus /> },
];

const Navbar: React.FC<NavbarProps> = ({ activePage }) => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-9999 bg-white shadow-md px-6 py-4 flex justify-between items-center">
      <div className="text-xl font-bold text-gray-700">
        Benchmarking Dashboard
      </div>
      <ul className="flex space-x-6">
        {navItems.map((item) => (
          <li key={item.name}>
            <a
              href={`/${item.name}`}
              className={twMerge(
                "flex items-center space-x-2 text-gray-600 hover:text-blue-600 transition-colors",
                activePage === item.name ? "font-semibold text-blue-600" : ""
              )}
            >
              {item.icon}
              <span>{item.label}</span>
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
};

export default Navbar;

import React from 'react';
import { FaLinkedinIn, FaGithub } from 'react-icons/fa'; // Added icons for sophistication

// Define colors for theme consistency
const DARK_BG = "#101828"; // Dark theme background
const ACCENT_COLOR = "#00C6FF"; // Primary blue accent
const LIGHT_TEXT = "#A0D4FF"; // Soft light text

export default function Footer() {
  return (
    // Removed 'fixed bottom-0' and used the dark theme color
    <footer className={`bg-[${DARK_BG}] text-gray-400 py-10 px-6 lg:px-20 w-full mt-16 shadow-inner shadow-black/50`}>
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center space-y-6 md:space-y-0">
        
        {/* 1. Copyright and Logo Link */}
        <div className="flex flex-col md:flex-row items-center space-y-2 md:space-y-0 md:space-x-4">
          <span className="text-xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-white to-[#00C6FF]">
            SyDev
          </span>
          <span className="text-sm border-l border-gray-700 md:pl-4">
            Copyright © {new Date().getFullYear()}. All rights reserved.
          </span>
        </div>

        {/* 2. Developer Credit and Social Icons (Modern Touch) */}
        <div className="flex items-center space-x-6">
          
          {/* Developer Credit */}
          <a
            className="text-sm font-light hover:text-white transition-colors flex items-center"
            target="_blank"
            rel="noopener noreferrer"
            href="https://www.linkedin.com/in/muhammed-naser-edden/"
          >
            Designed & Developed by{" "}
            <strong className={`text-[${LIGHT_TEXT}] ml-1 font-semibold hover:text-[${ACCENT_COLOR}]`}>
              Muhammed Nasser Edden
            </strong>
          </a>

          {/* Social Icons (Placeholder, assuming common developer platforms) */}
          <div className="space-x-4 hidden sm:flex">
            <a 
              href="https://www.linkedin.com/in/muhammed-naser-edden/"
              target="_blank" 
              rel="noopener noreferrer"
              className={`text-gray-500 hover:text-[${ACCENT_COLOR}] transition-colors`}
              aria-label="Developer LinkedIn"
            >
              <FaLinkedinIn size={20} />
            </a>
            <a 
              href="https://github.com/mhamdNaser" // Replace with actual GitHub link
              target="_blank" 
              rel="noopener noreferrer"
              className={`text-gray-500 hover:text-[${ACCENT_COLOR}] transition-colors`}
              aria-label="Developer GitHub"
            >
              <FaGithub size={20} />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
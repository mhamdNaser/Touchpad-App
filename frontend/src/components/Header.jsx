import React from "react";

export default function Header() {

  return (
    <header
      className={`py-4 lg:py-2 w-full mx-auto flex justify-between items-center z-50 transition-all duration-300 ease-in-out px-8 shadow-2xl bg-[#101828]`}
    >
      {/* Logo Section */}
      <div className="flex justify-between items-center w-full text-white">
        <div className="flex gap-3">
          <img
            src="/images/favicon.png"
            alt="Logo"
            className="max-h-10 lg:max-h-12 object-contain"
          />
          <h1 className="text-3xl lg:text-4xl font-extrabold font-sans tracking-widest text-transparent bg-clip-text bg-gradient-to-r from-white to-[#00C6FF]">
            SyDev
          </h1>
        </div>
        <div>AUT Group</div>
      </div>
    </header>
  );
}

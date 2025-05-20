import { createContext, useState, useMemo } from "react";
import { createTheme } from "@mui/material/styles";

// color design tokens export
export const tokens = (mode) => ({
  ...(mode === "dark"
    ? {
        grey: {
          100: "#FFFFFF",
          200: "#f4d2c1",
          300: "#ebbbab",
          400: "#e3a594",
          500: "#db8e7e",
          600: "#b27265",
          700: "#88554c",
          800: "#5e3933",
          900: "#2f1c19",
        },
        black: {
          10: "#FFFFFF",
          100: "#FFFFFF",
          200: "#FFF3E2",
        },
        white: {
          100: "#000000",
          200: "#202224",
        },
        red: {
          10: "#FFC3A1", //Casa
          20: "#FFC3A1",
        },
        yellow: {
          10: "#202224",
          20: "linear-gradient(180deg, #BFBFBF 0%, #202224 100%)",
          30: "#D9D9D9C9",
          40: "#F0D03A",
        },
        green: {
          10: "#B3C9F2",
        },
        primary: {
          10: "#000000",
          100: "#FFC3A1",
          200: "#fac3ac",
          300: "#f7a182",
          400: "#f89862",
          500: "#e87c50",
          600: "#c26841",
          700: "#9b5333",
          800: "#743f24",
          900: "#4d2a16",
          1000: "#000000",
        },

        greenAccent: {
          100: "#fff5dc",
          200: "#ffebb8",
          300: "#ffe194",
          400: "#f7c978",
          500: "#dbab5c",
          600: "#b28a4a",
          700: "#886838",
          800: "#5e4725",
          900: "#2f2513",
        },
        redAccent: {
          100: "#fde3df",
          200: "#fbc6bf",
          300: "#f7aa9f",
          400: "#e8583a",
          500: "#d14032",
          600: "#a73328",
          700: "#7e271e",
          800: "#541a14",
          900: "#2a0d0a",
        },
        blueAccent: {
          10: "#B3C9F2",
          100: "#eef8ff",
          200: "#dcefff",
          300: "#caf7ff",
          400: "#b8e6f7",
          500: "#a6d6ef",
          600: "#86acc3",
          700: "#678297",
          800: "#47586b",
          900: "#282e3e",
        },
      }
    : {
        grey: {
          100: "#4B4747",
          200: "#5e3933",
          300: "#88554c",
          400: "#b27265",
          500: "#db8e7e",
          600: "#e3a594",
          700: "#ebbbab",
          800: "#f4d2c1",
          900: "#fce8db",
        },
        black: {
          10: "#000000",
          100: "#4B4747",
          200: "#000000",
        },
        white: {
          100: "#FFFFFF",
          200: "#FFFFFF",
        },
        red: {
          10: "#7D0A0A", //Casa
          20: "#B8001F", //Đường
        },
        yellow: {
          10: "#FFF3E2",
          20: "linear-gradient(180deg, #FFE5CA 0%, #FFFFFF 100%)", //Guest activity
          30: "#FFE5CA", //Table header
          40: "#F0D03A", //pie chat
        },
        green: {
          10: "#507687",
        },
        primary: {
          10: "#FFFFFF",
          100: "#A75D5D",
          200: "#FFC3A1",
          300: "#D57373",
          400: "#E87D7D",
          500: "#E87C50",
          600: "#F89862",
          700: "#F7A182",
          800: "#FAC3AC",
          900: "#FDE4D7",
          1000: "#FFF3E2",
        },
        greenAccent: {
          100: "#FFF5DC",
          200: "#FFEBB8",
          300: "#FFE194",
          400: "#F7C978",
          500: "#DBAB5C",
          600: "#B28A4A",
          700: "#886838",
          800: "#5E4725",
          900: "#2F2513",
        },
        redAccent: {
          100: "#FDE3DF",
          200: "#FBC6BF",
          300: "#F7AA9F",
          400: "#E8583A",
          500: "#D14032",
          600: "#A73328",
          700: "#7E271E",
          800: "#541A14",
          900: "#2A0D0A",
        },
        blueAccent: {
          10: "#384B70",
          100: "#EEF8FF",
          200: "#DCEFFF",
          300: "#CAF7FF",
          400: "#B8E6F7",
          500: "#A6D6EF",
          600: "#86ACC3",
          700: "#678297",
          800: "#47586B",
          900: "#282E3E",
        },
      }),
});

// mui theme settings
export const themeSettings = (mode) => {
  const colors = tokens(mode);

  const cssVariables = Object.entries(colors).reduce((acc, [key, value]) => {
    Object.entries(value).forEach(([level, color]) => {
      acc[`--${key}-${level}`] = color;
    });
    return acc;
  }, {});

  return {
    palette: {
      mode: mode,
      ...(mode === "dark"
        ? {
            primary: { main: tokens(mode).primary[100] },
            secondary: { main: tokens(mode).greenAccent[400] },
            background: { default: "#000000" },
            text: {
              primary: "#FFFFFF",
            },
          }
        : {
            primary: { main: tokens(mode).primary[400] },
            secondary: { main: tokens(mode).greenAccent[400] },
            background: { default: "#FFFFFF" },
            text: {
              primary: "#000000",
            },
          }),
    },
    typography: {
      fontFamily: ["Montserrat", "sans-serif"].join(","),
      fontSize: 14,
      h1: {
        fontFamily: ["Montserrat", "sans-serif"].join(","),
        fontSize: 40,
      },
      h2: {
        fontFamily: ["Montserrat", "sans-serif"].join(","),
        fontSize: 32,
      },
      h3: {
        fontFamily: ["Montserrat", "sans-serif"].join(","),
        fontSize: 24,
      },
      h4: {
        fontFamily: ["Montserrat", "sans-serif"].join(","),
        fontSize: 20,
      },
      h5: {
        fontFamily: ["Montserrat", "sans-serif"].join(","),
        fontSize: 16,
      },
      h6: {
        fontFamily: ["Montserrat", "sans-serif"].join(","),
        fontSize: 14,
      },
      cssVariables,
    },
  };
};

// context for color mode
export const ColorModeContext = createContext({
  toggleColorMode: () => {},
});

export const useMode = () => {
  const [mode, setMode] = useState("light");

  const colorMode = useMemo(
    () => ({
      toggleColorMode: () =>
        setMode((prev) => (prev === "light" ? "dark" : "light")),
    }),
    []
  );

  const theme = useMemo(() => createTheme(themeSettings(mode)), [mode]);
  return [theme, colorMode];
};

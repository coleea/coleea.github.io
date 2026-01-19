import type {
  ExpressiveCodeConfig,
  LicenseConfig,
  NavBarConfig,
  ProfileConfig,
  SiteConfig,
} from "./types/config";
import { LinkPreset } from "./types/config";

export const siteConfig: SiteConfig = {
  title: "blog",
  subtitle: "subtitle",
  themeColor: {
    hue: 275, // Default hue for the theme color, from 0 to 360. e.g. red: 0, teal: 200, cyan: 250, pink: 345
    fixed: false, // Hide the theme color picker for visitors
  },
  banner: {
    enable: false,
    src: "./assets/images/Celestia.webp", // Relative to the /src directory. Relative to the /public directory if it starts with '/'
    // src: "./assets/images/demo-banner.DFyx781H_Z1SsDuN.webp", // Relative to the /src directory. Relative to the /public directory if it starts with '/'
    // src: "./assets/images/demo-banner.png", // Relative to the /src directory. Relative to the /public directory if it starts with '/'
    // src: "./assets/images/featured-sicp.jpg", // Relative to the /src directory. Relative to the /public directory if it starts with '/'
    // src: "./assets/images/sicp_wide.jpg", // Relative to the /src directory. Relative to the /public directory if it starts with '/'
    position: "center", // Equivalent to object-position, only supports 'top', 'center', 'bottom'. 'center' by default
    credit: {
      enable: false, // Display the credit text of the banner image
      text: "Pixiv @chokei", // Credit text to be displayed
      url: "https://www.pixiv.net/artworks/122782209", // (Optional) URL link to the original artwork or artist's page
    },
  },
  toc: {
    enable: true, // Display the table of contents on the right side of the post
    depth: 2, // Maximum heading depth to show in the table, from 1 to 3
  },
  favicon: [
    {
      src: "/favicon/icon.webp",
      //   theme: 'light',
      //   sizes: '32x32',
    },
  ],
};

export const navBarConfig: NavBarConfig = {
  links: [
    // LinkPreset.Home,
    // LinkPreset.Archive,
    // LinkPreset.Series,
    // LinkPreset.About,
    // LinkPreset.Friends,
    /*{
			name: "GitHub",
			url: "https://github.com/yCENzh", // Internal links should not include the base path, as it is automatically added
			external: true, // Show an external link icon and will open in a new tab
		},*/
    // {
    //   name: "여행",
    //   url: "https://www.travellings.cn/train.html",
    //   external: true,
    // },
  ],
};

export const profileConfig: ProfileConfig = {
  // avatar: "./assets/images/avatar.webp", // Relative to the /src directory. Relative to the /public directory if it starts with '/' 원assets/images/demo-avatar.png,선택https://q2.qlogo.cn/headimg_dl?dst_uin=189563385&spec=0
  // avatar: "./assets/images/avatar3.webp", // Relative to the /src directory. Relative to the /public directory if it starts with '/' 원assets/images/demo-avatar.png,선택https://q2.qlogo.cn/headimg_dl?dst_uin=189563385&spec=0
  avatar: "./assets/images/avatar_twitter_default_profile_400x400.png", // Relative to the /src directory. Relative to the /public directory if it starts with '/' 원assets/images/demo-avatar.png,선택https://q2.qlogo.cn/headimg_dl?dst_uin=189563385&spec=0  
  // avatar: "",
  name: "leekb",
  bio: "bio",
  links: [
    /*{
			name: "Twitter",
			icon: "fa6-brands:twitter", // Visit https://icones.js.org/ for icon codes
			// You will need to install the corresponding icon set if it's not already included
			// `pnpm add @iconify-json/<icon-set-name>`
			url: "https://twitter.com",
		},
		{
			name: "Steam",
			icon: "fa6-brands:steam",
			url: "https://store.steampowered.com",
		},*/
    {
      name: "GitHub",
      icon: "fa6-brands:github",
      url: "https://github.com/coleea",
    },
    // {
    //   name: "Telegram",
    //   icon: "fa6-brands:telegram",
    //   url: "https://t.me/yCENzh",
    // },
    // {
    //   name: "여행",
    //   icon: "fa6-solid:train-subway",
    //   url: "https://www.travellings.cn/train.html",
    // },
  ],
};

export const licenseConfig: LicenseConfig = {
  enable: true,
  name: "CC BY-NC-SA 4.0",
  url: "https://creativecommons.org/licenses/by-nc-sa/4.0/",
};

export const expressiveCodeConfig: ExpressiveCodeConfig = {
  // Note: Some styles (such as background color) are being overridden, see the astro.config.mjs file.
  // Please select a dark theme, as this blog theme currently only supports dark background color
  theme: "github-dark",
};

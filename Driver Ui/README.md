# RouteSense (India)# Welcome to your Lovable project



**Smart route planning for Indian roads with real-time hazard detection**## Project info



[![Built with Leaflet](https://img.shields.io/badge/Built%20with-Leaflet-199900?logo=leaflet)](https://leafletjs.com/)**URL**: https://lovable.dev/projects/040bcdd9-e0be-434e-8b90-b667085468af

[![OSRM](https://img.shields.io/badge/Routing-OSRM-5B9BD5?logo=openstreetmap)](http://project-osrm.org/)

[![Nominatim](https://img.shields.io/badge/Geocoding-Nominatim-7EBC6F?logo=openstreetmap)](https://nominatim.org/)## How can I edit this code?

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

There are several ways of editing your application.

---

**Use Lovable**

## 🚀 Live Demo

Simply visit the [Lovable Project](https://lovable.dev/projects/040bcdd9-e0be-434e-8b90-b667085468af) and start prompting.

**[Try RouteSense Live →](https://navi-sense-app.lovable.app)**

Changes made via Lovable will be committed automatically to this repo.

Experience intelligent routing with hazard detection for Indian roads.

**Use your preferred IDE**

---

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

## 📸 Screenshots

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

### Route selection with hazard legend

![Route Selection](docs/screenshot-routes.png)Follow these steps:

*Multiple route alternatives with traffic indicators and interactive hazard markers*

```sh

### Navigation mode with step list# Step 1: Clone the repository using the project's Git URL.

![Navigation Mode](docs/screenshot-navigation.png)git clone <YOUR_GIT_URL>

*Turn-by-turn directions with simulated movement and proximity alerts*

# Step 2: Navigate to the project directory.

> **Note**: Place screenshot images in the `docs/` folder at the repository root.cd <YOUR_PROJECT_NAME>



---# Step 3: Install the necessary dependencies.

npm i

## ✨ Features

# Step 4: Start the development server with auto-reloading and an instant preview.

### 🗺️ **Google-Maps-Style Interface**npm run dev

- Clean map background with right-side route cards```

- Smooth, road-snapped polylines with rounded caps and joins

- Interactive map controls with zoom, pan, and inertia**Edit a file directly in GitHub**



### 🇮🇳 **India-Focused Routing**- Navigate to the desired file(s).

- **Geocoding**: Nominatim with `countrycodes=IN` for accurate Indian location search- Click the "Edit" button (pencil icon) at the top right of the file view.

- **Routing**: OSRM with `polyline6` encoding for precise road-snapped routes- Make your changes and commit the changes.

- **Default view**: Centered on India [22.9734, 78.6569] at zoom level 5

- **2-4 route alternatives**: Automatically displays multiple options sorted by ETA**Use GitHub Codespaces**

- **Route D fallback**: For Mumbai→Pune, synthesizes a scenic route via Lonavala when OSRM returns <4 alternatives

- Navigate to the main page of your repository.

### 🎯 **Interactive Route Cards**- Click on the "Code" button (green button) near the top right.

- **Preview**: Highlights selected route and fits map bounds with smooth animation- Select the "Codespaces" tab.

- **Start**: Enters Navigation mode with simulated movement (40-60 km/h)- Click on "New codespace" to launch a new Codespace environment.

- **Traffic indicators**: Light/Moderate/Heavy badges based on relative duration- Edit files directly within the Codespace and commit and push your changes once you're done.

- **Route details**: ETA, distance, traffic, and contextual callouts

## What technologies are used for this project?

### ⚠️ **Intelligent Hazard Detection**

- **Three hazard types**:This project is built with:

  - 🔵 **Potholes** (blue circular markers)

  - 🟠 **Speed bumps** (orange rectangular markers)- Vite

  - 🟣 **Debris** (purple triangular markers)- TypeScript

- **Density-based placement**: Calculated per 100km (6/5/4 hazards respectively)- React

- **Adjustable density slider**: 0.5×, 1×, 1.5× (default), 2×- shadcn-ui

- **Smart distribution**: Evenly spaced along route with vertex snapping- Tailwind CSS

- **Interactive legend**: Toggle visibility per hazard type

- **Proximity alerts**: Toast notifications within ~120m during navigation## How can I deploy this project?

- **Detailed popups**: Hazard type, severity, distance from start, distance to destination

Simply open [Lovable](https://lovable.dev/projects/040bcdd9-e0be-434e-8b90-b667085468af) and click on Share -> Publish.

### 🎨 **Polished UX**

- **Smooth animations**: Snake-style route preview with 700ms draw animation## Can I connect a custom domain to my Lovable project?

- **Hover effects**: Route cards highlight corresponding map polyline

- **Draggable markers**: Adjust start/destination points with auto-rerouteYes, you can!

- **Accessible design**: Keyboard navigation, ARIA labels, focus states

- **Responsive layout**: Desktop-optimized with mobile-friendly drawerTo connect a domain, navigate to Project > Settings > Domains and click Connect Domain.



---Read more here: [Setting up a custom domain](https://docs.lovable.dev/features/custom-domain#custom-domain)


## 🛠️ Tech Stack

### Frontend
- **React** - UI framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - Accessible component library
- **Lucide React** - Icon system

### Mapping & Data
- **Leaflet** - Interactive maps with `preferCanvas: true` for performance
- **OpenStreetMap** - Tile layer (`https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png`)
- **OSRM** - Route calculation (public API)
  - Endpoint: `https://router.project-osrm.org/route/v1/driving/`
  - Parameters: `alternatives=3&overview=full&geometries=polyline6&steps=true`
- **Nominatim** - Geocoding with India bias
  - Endpoint: `https://nominatim.openstreetmap.org/search`
  - Parameters: `format=json&addressdetails=1&countrycodes=IN&limit=5`
- **@mapbox/polyline** - Polyline6 decoding

### Build & Deploy
- **Vite** - Fast build tool and dev server
- **Lovable** - Deployment platform

---

## 🚀 Getting Started (Local)

### Prerequisites
- **Node.js**: v18+ recommended
- **Package manager**: npm, pnpm, or yarn

### Installation & Development

```bash
# 1) Clone the repository
git clone https://github.com/Achintya1800/navi-sense-app.git
cd navi-sense-app

# 2) Install dependencies
npm install --legacy-peer-deps
# or
pnpm install
# or
yarn install

# 3) Start development server
npm run dev
# or
pnpm dev
# or
yarn dev

# App will be available at http://localhost:8081 (or next available port)
```

### Build for Production

```bash
# Build optimized bundle
npm run build

# Preview production build locally
npm run preview
```

### ⚠️ API Rate Limits

**No API keys required!** This app uses free, community-run services:

- **OSRM**: ~5 requests/second (fair use)
- **Nominatim**: 1 request/second per IP (strict)

For production use, consider:
- Self-hosting OSRM ([instructions](https://github.com/Project-OSRM/osrm-backend))
- Using a commercial provider (Mapbox, HERE, Google Maps)
- Implementing request caching and debouncing (already included for geocoding)

---

## ⚙️ Configuration

### Map Defaults

```typescript
// src/components/RouteMap.tsx
const map = L.map(containerRef.current, {
  preferCanvas: true,
  inertia: true,
}).setView([22.9734, 78.6569], 5); // India center, zoom 5
```

### OSRM Routing

```typescript
// src/services/routing.ts
const url = `https://router.project-osrm.org/route/v1/driving/${startLon},${startLat};${endLon},${endLat}?alternatives=3&overview=full&geometries=polyline6&steps=true`;
```

### Nominatim Geocoding

```typescript
// src/services/geocoding.ts
const response = await fetch(
  `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(
    query
  )}&format=json&addressdetails=1&countrycodes=IN&limit=5`,
  {
    headers: {
      "User-Agent": "RouteSense/1.0",
    },
  }
);
```

### Hazard Density

```typescript
// src/utils/hazardGenerator.ts
const baseDensity = {
  pothole: 6,      // per 100km
  speed_bump: 5,   // per 100km
  debris: 4,       // per 100km
};

// Default multiplier: 1.5×
// Adjustable via slider: 0.5×, 1×, 1.5×, 2×
// Min: 3 per type, Max: 10 per type (prevents clutter)
```

### Route D Fallback (Mumbai→Pune via Lonavala)

```typescript
// src/services/routing.ts
if (routes.length < 4) {
  const viaLon = 73.405;
  const viaLat = 18.750; // Lonavala coordinates
  
  // Fetch: start → Lonavala → end
  // Stitch polylines and steps
  // Add as "Route D" with subtitle "via Lonavala"
}
```

---

## 🔧 How It Works

### 1. **Polyline Decoding & Rendering**
- OSRM returns routes as `polyline6` (precision 6) encoded strings
- Decoded to `[lat, lng][]` arrays using `@mapbox/polyline`
- Rendered as Leaflet polylines with:
  - `smoothFactor: 1.5` for curve smoothing
  - `lineCap: "round"` and `lineJoin: "round"` for rounded edges
  - `renderer: L.canvas()` for better performance with many points

### 2. **Cumulative Distance Calculation**
```typescript
// For hazard placement and "distance to destination"
const distances = [0];
for (let i = 1; i < polyline.length; i++) {
  const dist = haversineDistance(polyline[i-1], polyline[i]);
  distances.push(distances[i-1] + dist);
}
```

### 3. **Hazard Placement Algorithm**
- Calculate total route distance in km
- Determine hazard count: `(distanceKm / 100) × baseDensity × densityFactor`
- Generate placement fractions (e.g., 5%, 12%, 20%, ..., 88%)
- For each fraction:
  - Find nearest vertex on polyline
  - Apply jitter (±2-4 vertices) for realism
  - Snap to exact vertex coordinates
- Store in `hazardsByRoute[routeId]`

### 4. **Layer Group Management**
```typescript
// Three separate layer groups for toggle control
hazardLayers = {
  pothole: L.layerGroup().addTo(map),
  speed_bump: L.layerGroup().addTo(map),
  debris: L.layerGroup().addTo(map),
};

// Render only for selected route
// Show/hide entire type with single layer operation
```

### 5. **Navigation Mode**
- Simulates movement along polyline at ~40-60 km/h
- Interpolates marker position every 2 seconds
- Monitors distance to all hazards on active route
- Triggers toast notification when within ~120m
- Displays turn-by-turn steps from OSRM's `legs[].steps[]`

---

## 🌍 Environment & Limits

### OSRM Public Service
- **Fair use policy**: ~5 requests/second
- **No authentication required**
- **Coverage**: Worldwide road network
- **Update frequency**: Weekly OSM data updates

### Nominatim Public Service
- **Strict rate limit**: 1 request/second per IP
- **Usage policy**: Must include `User-Agent` header
- **Built-in protection**: 250ms debounce on autocomplete
- **Alternative**: Self-host Nominatim for production

### Switching to Self-Hosted or Commercial APIs

**OSRM (self-hosted)**:
```typescript
// Update endpoint in src/services/routing.ts
const url = `https://your-osrm-instance.com/route/v1/driving/...`;
```

**Commercial alternatives**:
- Mapbox Directions API (requires API key)
- Google Maps Directions API (requires API key + billing)
- HERE Routing API (requires API key)

---

## 🚀 Deployment

### Live Production Instance

**[https://navi-sense-app.lovable.app](https://navi-sense-app.lovable.app)** ✨

Deployed on **Lovable** with automatic builds from the `main` branch.

### Deploy Your Own

#### Lovable Platform
1. Connect your GitHub repository
2. Configure build settings:
   - **Build command**: `npm run build`
   - **Output directory**: `dist`
   - **Install command**: `npm install --legacy-peer-deps`
3. Deploy automatically on push to `main`

#### Vercel
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod

# Build settings:
# - Framework: Vite
# - Build Command: npm run build
# - Output Directory: dist
```

#### Netlify
```bash
# Build settings in netlify.toml
[build]
  command = "npm run build"
  publish = "dist"

[build.environment]
  NODE_VERSION = "18"
```

**No environment variables required** - all APIs are public and client-side.

---

## 🗺️ Roadmap

### Planned Features
- [ ] **Real traffic overlays** from external APIs
- [ ] **Live hazard ingestion** from user reports or municipal data
- [ ] **User-generated reports** with photo upload
- [ ] **Offline mode** with cached tiles and routes
- [ ] **Turn-by-turn voice navigation** with multilingual support
- [ ] **Route history** and saved favorites
- [ ] **Multi-stop waypoints** for complex journeys
- [ ] **Elevation profiles** for hilly routes
- [ ] **Weather integration** (rain, fog, visibility)
- [ ] **Public transport** integration (buses, metro)

### Community Requests
- Alternative map styles (satellite, terrain)
- Route preferences (avoid tolls, highways, ferries)
- Fuel cost estimation
- EV charging station markers
- Accessibility mode (screen reader optimizations)

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Development Workflow
1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with clear, focused commits
4. **Run formatter**: `npm run lint` (if configured)
5. **Test locally**: Verify routes work for Mumbai→Pune and other corridors
6. **Submit a Pull Request** with a clear description

### Code Style
- Use **TypeScript** for type safety
- Follow **existing patterns** for components and utilities
- Add **comments** for complex logic (especially routing/hazard algorithms)
- Keep components **focused and reusable**

### Branch Naming
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code improvements without behavior change

### Testing
- Test route generation for Mumbai→Pune (should show 4 routes)
- Verify hazard density slider (0.5×-2×) regenerates markers
- Confirm proximity alerts trigger during navigation
- Check responsive behavior on mobile viewports

---

## 📄 License

**MIT License**

Copyright (c) 2025 Achintya1800

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 🙏 Acknowledgements

This project is built on the shoulders of amazing open-source communities:

- **[OpenStreetMap Contributors](https://www.openstreetmap.org/copyright)** - For the world's best open geographic database
- **[OSRM (Open Source Routing Machine)](http://project-osrm.org/)** - For blazing-fast routing calculations
- **[Nominatim](https://nominatim.org/)** - For accurate geocoding worldwide
- **[Leaflet](https://leafletjs.com/)** - For the lightweight, powerful mapping library
- **[shadcn/ui](https://ui.shadcn.com/)** - For beautiful, accessible React components
- **[Tailwind CSS](https://tailwindcss.com/)** - For utility-first styling
- **[Lovable](https://lovable.app/)** - For seamless deployment and hosting

Special thanks to all contributors and users testing routes across India! 🇮🇳

---

## 📞 Support

- **Live Demo**: [https://navi-sense-app.lovable.app](https://navi-sense-app.lovable.app)
- **Issues**: [GitHub Issues](https://github.com/Achintya1800/navi-sense-app/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Achintya1800/navi-sense-app/discussions)

---

<div align="center">

**Made with ❤️ for Indian roads**

[⭐ Star this repo](https://github.com/Achintya1800/navi-sense-app) if you find it useful!

</div>

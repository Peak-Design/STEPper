# STEPper NEXT - STEP File Importer for Blender

Blender addon for importing STEP (`.step` / `.stp`) files directly into Blender using the OpenCASCADE (OCC) geometry kernel. The produced mesh is a triangulation of the underlying CAD surface with smooth normals computed from the analytic shape geometry.

Originally created by **ambi** (Tommi Hyppanen). Now maintained by **Peak Design**.

## Features

- Direct STEP file import via OpenCASCADE
- Analytic surface normals for smooth, seamless shading on curved surfaces
- Per-face vertex colors and automatic material creation from STEP color data
- Part hierarchy preserved as Blender object tree
- Robust handling of corrupted geometry - attempts to import everything it can instead of skipping entire parts
- ShapeFix healing for shapes with missing or damaged geometry
- Native C++ mesh extraction with multithreaded normal computation
- Up to 5x faster import speeds compared to v1.x

## Requirements

- **Windows 10+ (64-bit)** - Windows only for now
- **Blender 5.1** with Python 3.13
- Visual Studio C++ Redistributable: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170 (vc_redist.x64.exe)

## Installation

1. Download the latest release repo as a `.zip` file.
2. In Blender, go to **Edit > Preferences > Add-ons** and click **Install...**, then select the `.zip` file.
3. Enable the addon in the Add-ons list.

The importer panel will appear in **3D View > Tools panel > STEPper NEXT**.

## Uninstall / Update

Restart Blender, then remove the addon from Preferences > Add-ons. To update, remove the old version first, restart Blender, then install the new `.zip`.

## Version History

| Version | Blender | Changes |
|---------|---------|---------|
| 2.1.3   | 5.1     | Renamed to STEPper NEXT, auto-apply scale, skip empty objects, preferences now persist across sessions |
| 2.1.x   | 5.1     | Multithreaded normal computation, performance optimizations, crash fixes for corrupt STEP files |
| 2.1.0   | 5.1     | Updated to pythonocc-core 7.9.3 / Python 3.13, native C++ mesh extraction |
| 2.0.0   | 5.0     | Ported to Blender 5.0 API, added import diagnostics and failed parts reporting |
| 1.1.8   | 4.2.1   | Last release by ambi |

## Support

This addon is free and open source under the GPL v3 license.

**ambi** - original creator:
https://ambient.gumroad.com/l/stepper

**Peak Design** - current maintainer, tips welcome:
https://ko-fi.com/oskarasspalvys

## License

This program is free software under the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.html).

Copyright 2021 Tommi Hyppanen
Modified 2026 by Peak-Design

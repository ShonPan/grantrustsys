# Network Visualization — Claude Code Instructions

## What This Does

A force-directed graph showing the grant ecosystem: applicants, funders, events,
and mentors. Three toggleable views that tell the control protocol story visually.

This is the demo moment where judges SEE the difference between genuine and
spray-and-pray applicants at a glance.

## Three Views

### View 1: All Connections
Everything visible. Applicants connected to funders they applied to, events
they attended, mentors they worked with, and collaborators.

### View 2: Spray-and-Pray Cluster
Highlight spray applicants. They show:
- Dense connections to MANY funders (applied everywhere)
- ZERO event connections (never attended anything)
- ZERO mentor connections (no community ties)
- Stylometric similarity edges between their applications (same template)
- Empty reputation (no GitHub, no hackathons)
Visual: tight cluster of red nodes connected to every funder, isolated from
the community layer

### View 3: Genuine Cluster
Highlight genuine applicants. They show:
- Thin, targeted funder connections (applied to 1-2 matched funders)
- Rich event connections (hackathons, conferences, workshops)
- Mentor connections (people they've worked with)
- Collaboration edges (co-authored, co-built)
- Strong reputation signals
Visual: distributed nodes embedded in a community network

## Node Types and Colors

| Type | Color | Shape | Size |
|------|-------|-------|------|
| Genuine applicant | Cyan (#00e5ff) | Circle | Medium |
| Spray applicant | Red (#ff4444) | Circle | Medium |
| Rising applicant | Amber (#ffb800) | Circle | Medium |
| Funder | Green (#00d97e) | Diamond/Square | Large |
| Bad-faith funder | Red (#ff4444) | Diamond/Square | Large, dashed border |
| Event | Purple (#b388ff) | Triangle | Small |
| Mentor | White (#c8daf5) | Small circle | Small |

## Edge Types

| Type | Style | Color |
|------|-------|-------|
| Applied to funder | Solid line | Gray, thickness = fit score |
| Attended event | Dashed line | Purple |
| Mentored by | Dotted line | White, thin |
| Collaborated with | Solid line | Cyan |
| Stylometric similarity | Wavy/dashed red | Red (spray-to-spray only) |

## Data Source

Load from the existing data/applicants/*.json and data/funders/*.json files.
Build the graph edges from:

**Applicant → Funder edges:**
- Applicant A → Deep Science Ventures (fit: high)
- Applicant B → Deep Science Ventures (fit: low, spray)
- Applicant B → Responsible AI Foundation (fit: low, spray)
- Applicant B → Green Horizons (fit: low, spray)
- Applicant B → NexGen Biotech (fit: low, spray)
- Applicant C → Deep Science Ventures (fit: low, wrong funder)
- Applicant C → Green Horizons (fit: high, right funder)
- Applicant D → NexGen Biotech (fit: medium)

**Applicant → Event edges:**
- Applicant A → Frontier Tower Hackathon
- Applicant A → SpaceHack 2025
- Applicant B → (none)
- Applicant B_v2 → SpaceHack 2026
- Applicant C → ClimateHack 2025
- Applicant D → BioHack 2025

**Applicant → Collaborator edges:**
- Applicant A ↔ "Shon Pan" (Ground Truth)
- Applicant C ↔ "UrbanAir Collective"
- Applicant B_v2 ↔ "SpaceHack Team 12"

**Spray similarity edges (red):**
- Applicant B app_v1 ↔ Applicant B app_v2 (cosine similarity 0.95+)

**Bad-faith funder nodes (if they exist in data/funders/):**
- Theranos Innovation Fund — flagged with red flags
- Synergy AI Capital — flagged with red flags

## Implementation

Build as a standalone HTML file using D3.js force simulation.
Save to `src/network_viz.html` or `static/network.html`.

### Key D3 setup:

```javascript
// Force simulation
const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(edges).id(d => d.id).distance(d => {
        // Shorter distance for strong connections
        return d.type === 'applied' ? 150 - (d.fit_score * 80) : 120;
    }))
    .force("charge", d3.forceManyBody().strength(-200))
    .force("center", d3.forceCenter(width/2, height/2))
    .force("collision", d3.forceCollide().radius(25));
```

### View toggle buttons:

Three buttons at top: "All" | "Spray Pattern" | "Genuine Pattern"

Toggling views doesn't remove nodes — it changes opacity and highlights:
- "Spray Pattern": spray nodes + their edges at full opacity, everything else at 0.15
- "Genuine Pattern": genuine nodes + their edges at full opacity, everything else at 0.15
- "All": everything at full opacity

### Tooltip on hover:

Show node details on hover:
- Applicant: name, intent score, reputation score, number of events attended
- Funder: name, focus areas, number of applications received
- Event: name, date, domain

### Intent score badge on applicant nodes:

Each applicant node shows a small score label below it:
- "0.74 ✓" in green for genuine
- "0.43 ✗" in red for spray
- "0.61 ?" in amber for mixed

## Integration with Dashboard

Add a "Network" tab or link in the dashboard navigation that opens the
visualization. The visualization can be a separate page or embedded via iframe.

Add a route to the Flask app:

```python
@app.route("/network")
def network():
    return render_template_string(NETWORK_HTML)
```

Or serve as a static file.

## Dashboard Integration Alternative: Embed in results

Instead of a separate page, add a small version of the graph in the
dashboard results panel that shows the current applicant's position
in the network. Highlight their node and connections. This is more
compelling for live demo but harder to build.

Recommend: build as separate page first, embed later if time allows.

## Claude Code Command

```
Read network_viz_spec.md in the repo root. Create a network visualization
as an interactive HTML page using D3.js (load from CDN). The graph shows
applicants, funders, events, and mentors as a force-directed layout.

Build node and edge data from the existing JSON files in data/applicants/
and data/funders/. Add three toggle buttons for "All Connections",
"Spray Pattern", and "Genuine Pattern" views that change node/edge opacity
to highlight each cluster.

Use the Xenarch terminal aesthetic from the dashboard: dark background (#04060d),
cyan accent (#00e5ff), red for spray (#ff4444), green for funders (#00d97e).

Add hover tooltips showing node details. Show intent scores as badges
on applicant nodes.

Save to static/network.html and add a /network route to the Flask app
in grant_trust_dashboard.py that serves it.
```

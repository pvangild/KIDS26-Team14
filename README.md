# Biohackathon Project Template

This repository is a starting point for a three-day team project. This repository is populated with a starting template for team organization and planning. Use it to plan, build, and document work. Please adjust this repository to suit the needs of your team.

> **Team leads:** Start with the [team lead checklist](project-management/CHECKLIST.md) before the event or during your first team meeting.

## Project Profile

- **Project name:** [Add a short, descriptive name]
- **Question, problem, or opportunity:** [What are you exploring?]
- **Data, inputs, or evidence:** [What will you use, and where does it come from?]
- **Expected output:** [What will you show, test, explain, or demonstrate?]
- **Tools and stack:** [Languages, libraries, notebooks, APIs, databases, services, or other tools]
- **Team lead:** [Name and GitHub handle]
- **Team members and roles:** [Link to `project-management/team.md`]
- **Communication:** [Add the agreed channel or contact]

Naming the tools and stack early helps the team lead create useful roles and divide work realistically. It is fine to revise this section as the project develops.

## Vision and Mission

- **Vision:** [Describe the change, insight, or capability you hope this project supports.]
- **Mission:** [Describe what the team will do during the biohackathon to move toward that vision.]

## About

[Add a short explanation of the motivation, background, and why the question or problem matters.]

## Analyze Recordings

`analyze_recordings.py` reads every `.mat` recording in a folder and writes a Markdown event-count table. It requires Python with `numpy`, `scipy`, and `matplotlib` installed.

```powershell
python analyze_recordings.py --data-dir "Z:\path\to\recordings"
```

To plot a particular event, choose files and events using 1-based numbering after sorting files alphabetically. For example, this plots apnea event 3 from the second file:

```powershell
python analyze_recordings.py --data-dir "Z:\path\to\recordings" --plot-apnea --file 2 --event 3
```

Use `--plot-hypopnea` to plot a hypopnea instead. By default, the summary is saved as `event_summary.md`; change that path with `--summary-file`.

## Roadmap and Milestones

| When | Focus | Expected outcome |
| --- | --- | --- |
| Day 1 | Agree on the question, inputs, stack, roles, and first tasks | A shared plan and a first small change in the repository |
| Day 2 | Build, test, and compare approaches | A working result or clear evidence about what does not work |
| Day 3 | Stabilize, document, and present | A demo or handoff with methods, limitations, and next steps |

The goal is not a perfect production system. The goal is a clear, honest, useful result that the team can explain and others can build on.



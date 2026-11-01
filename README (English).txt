audiospace 0.7.0 - README (English)

audiospace is a 3D audio component (a library) for games and tools for blind and visually impaired people, meant to be heard on headphones.
It comes with the Localization Tool, which lets you check which direction sounds come from and how rooms sound. Both work with the keyboard and a screen reader (NVDA, PC-Talker) alone.
The tool shows English when the display language of Windows is not Japanese.


== What is in this folder

There are 16 files and 5 folders.

- README (English).txt is this file.
- Manual (English).html and Manual (English).txt explain how to use it. They have the same content. The html file can be read in a web browser, moving by headings; the txt file can be read from top to bottom in Notepad.
- Start Localization Tool.bat starts the Localization Tool. Python and the components must be installed first (see "First three steps" below).
- README.en.md is a detailed guide for people who use the library in a game.
- CUES.en.md lists the processing that helps you tell front from back and up from down.
- LICENSE is the full text of the license (MIT).
- ATTRIBUTION.en.md lists where the included data comes from and the credits you must show.
- pyproject.toml is the definition file used when installing the components.
- The following files are the Japanese editions of the files above: はじめにお読みください.txt (this file), マニュアル.html and マニュアル.txt (the manual), 定位確認ツールを起動.bat (the same launcher), README.md, CUES.md and ATTRIBUTION.md.
- The audiospace folder is the library itself.
- The assets folder has the measured head-related transfer functions (sound data for each ear shape) and computed reverbs.
- The demo folder is the Localization Tool itself.
- The tests folder has tests that check that everything works.
- The tools folder has tools for rebuilding the data and documents. You do not need it to use audiospace.

When you use the Localization Tool, two more folders appear here: logs (your listening results) and settings (the spaces and reverbs you design).


== First three steps

1. Put this folder in a folder you can write to, such as Documents. Do not put it inside Program Files.
2. Install Python 3.10 or later, open a command prompt in this folder and run this line:
   python -m pip install -e ".[app]"
   If it says that python cannot be found, run it with py instead of python at the start.
   Chapter 3 of the manual, "Setting up", has the details.
3. Open Start Localization Tool.bat. When a window called "Localization Tool - audiospace" opens with the keyboard help in it, you are ready. F1 opens the keyboard help in a window of its own.

Use headphones. With speakers, the left and right sounds mix, so directions cannot be heard.


== Quitting

Press Escape to quit the Localization Tool.


That is all.

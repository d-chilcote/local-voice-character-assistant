# Project TODO List

## Features to Implement

### Sound Effects System
- [ ] Fix async sound playback using macOS `afplay` command
- [ ] Test sound effect triggers: `[BURP]`, `[LAUGH]`, `[CLANK]`, `[DRINK]`
- [ ] Map tags to appropriate system sounds in `/System/Library/Sounds/`
- [ ] Ensure sound playback doesn't block TTS generation
- [ ] Add sound effects back to Bender's capabilities in system prompt

### Future Enhancements
- [ ] Add `/reset` endpoint documentation to README
- [ ] Consider adding more character voices/personalities
- [ ] Explore streaming TTS for lower latency
- [ ] Add conversation export functionality
- [ ] Implement better error handling for Gemini API failures

import logging
import json
import random
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class GameMasterAgent(Agent):
    """D&D-style voice Game Master for interactive adventures"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are an epic Game Master running a fantasy adventure in the world of Eldoria, a realm of magic, dragons, and ancient mysteries.

YOUR ROLE AS GAME MASTER:
You narrate an immersive, interactive story where the player is the hero. You describe scenes vividly, present challenges, and respond to player actions with engaging outcomes.

UNIVERSE & SETTING:
- **World**: Eldoria - a fantasy realm with kingdoms, dark forests, ancient ruins, and mystical creatures
- **Tone**: Epic and adventurous, with moments of danger, mystery, and triumph
- **Magic**: Exists but is rare and powerful
- **Creatures**: Dragons, goblins, wizards, enchanted beasts

YOUR STORYTELLING STYLE:
- Start by introducing the player as a brave adventurer
- Paint vivid scenes with rich descriptions
- Present clear choices or ask "What do you do?"
- React dynamically to player decisions
- Build tension and excitement
- Remember what happened earlier in the story
- Keep the adventure moving forward

STORY STRUCTURE:
1. **Opening**: Set the scene, introduce the quest
2. **Challenge**: Present obstacles, enemies, or puzzles
3. **Climax**: Build to an exciting moment
4. **Resolution**: Conclude the current mini-arc

IMPORTANT RULES:
- Always end your turn by asking what the player does next
- Keep responses under 4 sentences when describing scenes
- Be dramatic but not overwhelming
- Accept creative solutions from the player
- Make the player feel heroic
- Track key events and reference them later
- No emojis or special formatting

START THE ADVENTURE:
Begin by asking the player their name, then introduce them to the world of Eldoria with an exciting opening scenario. Maybe they're in a tavern hearing about a mysterious quest, or they wake up in a dark forest with no memory, or they're approached by a desperate villager seeking help. Make it engaging!"""
        )
        
        # Game state tracking
        self.game_state = {
            "player_name": "",
            "current_location": "Starting Point",
            "inventory": [],
            "health": "Healthy",
            "key_events": [],
            "npcs_met": [],
            "quest_status": "Starting new adventure",
            "turn_count": 0
        }
    
    @function_tool
    async def update_player_info(self, context: RunContext, player_name: str = "", 
                                 location: str = "", health: str = "") -> str:
        """Update player information
        
        Args:
            player_name: The player's character name
            location: Current location in the game world
            health: Player's health status (Healthy, Injured, Critical)
        """
        if player_name:
            self.game_state["player_name"] = player_name
            logger.info(f"Player name set to: {player_name}")
        
        if location:
            self.game_state["current_location"] = location
            logger.info(f"Location changed to: {location}")
        
        if health:
            self.game_state["health"] = health
            logger.info(f"Health updated to: {health}")
        
        return "Player info updated"
    
    @function_tool
    async def add_to_inventory(self, context: RunContext, item: str) -> str:
        """Add an item to player's inventory
        
        Args:
            item: The item to add (e.g., "magic sword", "health potion", "ancient key")
        """
        self.game_state["inventory"].append(item)
        logger.info(f"Added to inventory: {item}")
        return f"{item} added to inventory"
    
    @function_tool
    async def remove_from_inventory(self, context: RunContext, item: str) -> str:
        """Remove an item from player's inventory
        
        Args:
            item: The item to remove
        """
        if item in self.game_state["inventory"]:
            self.game_state["inventory"].remove(item)
            logger.info(f"Removed from inventory: {item}")
            return f"{item} removed from inventory"
        return f"{item} not found in inventory"
    
    @function_tool
    async def record_event(self, context: RunContext, event: str) -> str:
        """Record a significant story event
        
        Args:
            event: Description of the event (e.g., "Defeated the goblin chief", "Found the ancient map")
        """
        self.game_state["key_events"].append({
            "turn": self.game_state["turn_count"],
            "event": event
        })
        logger.info(f"Recorded event: {event}")
        return "Event recorded"
    
    @function_tool
    async def meet_npc(self, context: RunContext, npc_name: str, npc_role: str = "") -> str:
        """Record meeting an NPC
        
        Args:
            npc_name: The NPC's name
            npc_role: Their role (e.g., "wizard", "merchant", "village elder")
        """
        npc = {"name": npc_name, "role": npc_role}
        self.game_state["npcs_met"].append(npc)
        logger.info(f"Met NPC: {npc_name} ({npc_role})")
        return f"Recorded meeting with {npc_name}"
    
    @function_tool
    async def check_inventory(self, context: RunContext) -> str:
        """Check what's in the player's inventory"""
        if not self.game_state["inventory"]:
            return "Your inventory is empty."
        
        items = ", ".join(self.game_state["inventory"])
        return f"You are carrying: {items}"
    
    @function_tool
    async def roll_dice(self, context: RunContext, dice_type: int = 20, modifier: int = 0) -> str:
        """Roll dice for skill checks or combat
        
        Args:
            dice_type: Type of dice (6, 12, 20, etc.)
            modifier: Bonus or penalty to add
        """
        roll = random.randint(1, dice_type)
        total = roll + modifier
        
        # Determine success level
        if total >= 15:
            result = "Critical Success!"
        elif total >= 10:
            result = "Success"
        elif total >= 5:
            result = "Partial Success"
        else:
            result = "Failure"
        
        logger.info(f"Dice roll: d{dice_type} = {roll} + {modifier} = {total} ({result})")
        
        return f"You rolled {roll} (+ {modifier} modifier) = {total}. {result}!"
    
    @function_tool
    async def show_game_status(self, context: RunContext) -> str:
        """Show current game status"""
        status = f"Player: {self.game_state['player_name'] or 'Adventurer'}\n"
        status += f"Location: {self.game_state['current_location']}\n"
        status += f"Health: {self.game_state['health']}\n"
        status += f"Inventory: {', '.join(self.game_state['inventory']) if self.game_state['inventory'] else 'Empty'}\n"
        status += f"Turns: {self.game_state['turn_count']}"
        
        return status
    
    @function_tool
    async def save_game(self, context: RunContext) -> str:
        """Save the current game state to a file"""
        save_dir = Path("game_saves")
        save_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = save_dir / f"adventure_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.game_state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Game saved to {filename}")
        return f"Game saved! Your adventure has been preserved."
    
    def _increment_turn(self):
        """Increment turn counter"""
        self.game_state["turn_count"] += 1


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",  # Deep, storytelling voice
            style="Narration",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
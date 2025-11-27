import logging
import json
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


def load_company_faq():
    """Load company FAQ content"""
    faq_file = Path("shared-data/razorpay_faq.json")
    if faq_file.exists():
        with open(faq_file, "r") as f:
            return json.load(f)
    else:
        logger.warning(f"FAQ file not found: {faq_file}")
        return {}


COMPANY_FAQ = load_company_faq()


class SDRAgent(Agent):
    """Sales Development Representative agent"""
    
    def __init__(self) -> None:
        company_name = COMPANY_FAQ.get("company_name", "our company")
        product = COMPANY_FAQ.get("product", "our product")
        
        super().__init__(
            instructions=f"""You are Priya, a friendly and professional Sales Development Representative (SDR) for {company_name}.

YOUR ROLE:
You help potential customers understand {product} and capture their information to follow up later.

CONVERSATION FLOW:
1. GREET warmly and introduce yourself
2. ASK what brought them here and what they're working on
3. LISTEN to their needs and answer questions using the FAQ
4. COLLECT lead information naturally during conversation
5. SUMMARIZE and confirm when they're ready to end

LEAD INFORMATION TO COLLECT (ask naturally, not like a form):
- Name
- Company name
- Email
- Role/Position
- What they want to use the product for (use case)
- Team size
- Timeline (now / soon / later / just exploring)

IMPORTANT GUIDELINES:
- Be conversational and warm, not robotic
- Answer questions using the search_faq tool - don't make up information
- Ask for lead details naturally throughout the conversation, not all at once
- When you detect they're done (e.g., "that's all", "thanks", "I'm good"), use save_lead to save their info
- Keep responses brief and natural - you're speaking out loud!
- No emojis or special formatting

ANSWERING QUESTIONS:
- When they ask about the product, pricing, features, etc., use search_faq tool
- Base your answers on the FAQ content
- If something isn't in the FAQ, say "That's a great question! Let me connect you with our team for detailed information on that"

Start by introducing yourself and asking what brought them here today!"""
        )
        
        # Track lead information
        self.lead_data = {
            "name": "",
            "company": "",
            "email": "",
            "role": "",
            "use_case": "",
            "team_size": "",
            "timeline": "",
            "questions_asked": [],
            "timestamp": ""
        }
    
    @function_tool
    def search_faq(self, context: RunContext, query: str) -> str:
        """Search the company FAQ for answers to customer questions
        
        Args:
            query: The customer's question or topic (e.g., 'pricing', 'features', 'who is this for')
        """
        logger.info(f"Searching FAQ for: {query}")
        
        query_lower = query.lower()
        
        # Track what they asked about
        if query not in self.lead_data["questions_asked"]:
            self.lead_data["questions_asked"].append(query)
        
        # Search through FAQs
        best_match = None
        best_score = 0
        
        for faq in COMPANY_FAQ.get("faqs", []):
            question = faq["question"].lower()
            answer = faq["answer"]
            
            # Simple keyword matching
            score = 0
            for word in query_lower.split():
                if len(word) > 3 and word in question:
                    score += 2
                if len(word) > 3 and word in answer.lower():
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = answer
        
        if best_match and best_score > 0:
            return best_match
        else:
            return "I don't have specific information about that in my FAQ. Let me connect you with our team who can provide detailed information!"
    
    @function_tool
    def update_lead_info(self, 
                        context: RunContext,
                        name: str = "",
                        company: str = "",
                        email: str = "",
                        role: str = "",
                        use_case: str = "",
                        team_size: str = "",
                        timeline: str = ""):
        """Update lead information as it's collected during conversation
        
        Call this whenever you learn new information about the lead.
        
        Args:
            name: Lead's full name
            company: Company name
            email: Email address
            role: Job role/position
            use_case: What they want to use the product for
            team_size: Size of their team (e.g., '5-10', 'just me', '50+')
            timeline: When they're looking to start (now / soon / later / exploring)
        """
        if name:
            self.lead_data["name"] = name
        if company:
            self.lead_data["company"] = company
        if email:
            self.lead_data["email"] = email
        if role:
            self.lead_data["role"] = role
        if use_case:
            self.lead_data["use_case"] = use_case
        if team_size:
            self.lead_data["team_size"] = team_size
        if timeline:
            self.lead_data["timeline"] = timeline
        
        logger.info(f"Updated lead info: {self.lead_data}")
        return "Information recorded!"
    
    @function_tool
    def save_lead(self, context: RunContext, summary: str = ""):
        """Save the complete lead information when the conversation ends
        
        Call this when the user indicates they're done (e.g., 'that's all', 'thanks', 'goodbye')
        
        Args:
            summary: Brief summary of the conversation and the lead's interest
        """
        logger.info("Saving lead information")
        
        # Add timestamp and summary
        self.lead_data["timestamp"] = datetime.now().isoformat()
        self.lead_data["conversation_summary"] = summary
        
        # Create leads directory
        leads_dir = Path("leads")
        leads_dir.mkdir(exist_ok=True)
        
        # Save to JSON file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        lead_name = self.lead_data.get("name", "unknown").replace(" ", "_")
        filename = leads_dir / f"lead_{lead_name}_{timestamp_str}.json"
        
        with open(filename, "w") as f:
            json.dump(self.lead_data, f, indent=2)
        
        logger.info(f"Lead saved to {filename}")
        
        # Create verbal summary
        name = self.lead_data.get("name", "the visitor")
        company = self.lead_data.get("company", "")
        use_case = self.lead_data.get("use_case", "")
        timeline = self.lead_data.get("timeline", "")
        
        verbal_summary = f"Perfect! I've recorded all your details, {name}"
        if company:
            verbal_summary += f" from {company}"
        verbal_summary += "."
        
        if use_case:
            verbal_summary += f" You're interested in using {COMPANY_FAQ.get('company_name', 'our product')} for {use_case}."
        
        if timeline and timeline.lower() != "exploring":
            verbal_summary += f" Your timeline is {timeline}."
        
        verbal_summary += " Our team will reach out to you shortly. Thanks for your time today!"
        
        return verbal_summary


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # Preload FAQ data
    proc.userdata["faq"] = load_company_faq()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-priya",  # Indian English female voice
            style="Conversation",
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
        agent=SDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
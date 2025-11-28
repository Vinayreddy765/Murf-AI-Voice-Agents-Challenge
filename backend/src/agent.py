import logging
import json
from pathlib import Path
from datetime import datetime

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


def load_fraud_cases():
    """Load fraud cases from JSON database"""
    db_file = Path("shared-data/fraud_cases.json")
    if db_file.exists():
        with open(db_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        logger.warning(f"Fraud cases database not found: {db_file}")
        return []


def save_fraud_cases(cases):
    """Save updated fraud cases back to database"""
    db_file = Path("shared-data/fraud_cases.json")
    with open(db_file, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    logger.info(f"Fraud cases database updated: {db_file}")


class FraudAlertAgent(Agent):
    """Bank fraud detection and verification agent"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a professional fraud detection representative from HDFC Bank's Security Department.

YOUR ROLE:
You are calling customers about suspicious transactions detected on their accounts. You need to verify their identity and confirm if transactions are legitimate.

CRITICAL SECURITY RULES:
- NEVER ask for full card numbers, PINs, passwords, or OTPs
- Only use the security question from the database for verification
- Be calm, professional, and reassuring
- Make it clear this is about their account security

CALL FLOW:
1. INTRODUCE yourself and the bank
2. ASK for the customer's name to look up their case
3. VERIFY identity using the security question from database
4. If verification FAILS → End call politely
5. If verification PASSES:
   - Read the suspicious transaction details
   - Ask if they made this transaction (yes/no)
   - Based on their answer, update the case status
6. EXPLAIN what action will be taken
7. Thank them and end the call

Keep responses brief, clear, and professional - you're speaking on a phone call.
No emojis or special formatting.

Start by introducing yourself and asking for their name!"""
        )
        
        # Load fraud cases database
        self.fraud_cases = load_fraud_cases()
        self.current_case = None
        self.verification_passed = False
    
    @function_tool
    async def load_user_case(self, context: RunContext, username: str) -> str:
        """Load the fraud case for a specific user
        
        Args:
            username: The customer's name to look up
        """
        logger.info(f"Looking up fraud case for user: {username}")
        
        username_lower = username.lower().strip()
        
        # Search for user in database
        for case in self.fraud_cases:
            if case["userName"].lower() == username_lower:
                self.current_case = case
                logger.info(f"Found fraud case for {username}: {case}")
                
                # Return confirmation (don't reveal transaction details yet)
                return f"Thank you, {case['userName']}. I have your account pulled up. For security purposes, I need to verify your identity before we proceed."
        
        # User not found
        return f"I apologize, but I don't have a fraud alert case for {username} in my system. Please call our customer service line if you believe this is an error."
    
    @function_tool
    async def verify_security_answer(self, context: RunContext, answer: str) -> str:
        """Verify the customer's answer to the security question
        
        Args:
            answer: The customer's answer to the security question
        """
        if not self.current_case:
            return "I need to look up your case first. Please tell me your name."
        
        logger.info(f"Verifying security answer: {answer}")
        
        correct_answer = self.current_case.get("securityAnswer", "").lower().strip()
        user_answer = answer.lower().strip()
        
        if user_answer == correct_answer:
            self.verification_passed = True
            logger.info("Security verification PASSED")
            
            # Read transaction details
            case = self.current_case
            details = f"Thank you for verifying. I can now share the details. We detected a suspicious transaction on your card ending in {case['cardEnding']}. "
            details += f"The transaction was for {case['transactionAmount']} at {case['transactionName']}, "
            details += f"made on {case['transactionTime']} from {case['transactionLocation']}. "
            details += f"The merchant category is {case['transactionCategory']}. Did you make this transaction?"
            
            return details
        else:
            self.verification_passed = False
            logger.warning("Security verification FAILED")
            return "I'm sorry, but that answer doesn't match our records. For your security, I cannot proceed with this call. Please visit your nearest branch or call our customer service line with proper identification. Thank you."
    
    @function_tool
    async def get_security_question(self, context: RunContext) -> str:
        """Get the security question for the current user
        
        Returns the security question that needs to be asked
        """
        if not self.current_case:
            return "I need to look up your case first. Please tell me your name."
        
        question = self.current_case.get("securityQuestion", "What is your date of birth?")
        return f"For security verification, please answer this question: {question}"
    
    @function_tool
    async def confirm_transaction_status(self, context: RunContext, customer_made_transaction: bool) -> str:
        """Record whether the customer confirms or denies making the transaction
        
        Args:
            customer_made_transaction: True if customer confirms they made it, False if they deny
        """
        if not self.current_case:
            return "I need to look up your case first."
        
        if not self.verification_passed:
            return "I cannot proceed without proper security verification."
        
        logger.info(f"Customer confirmed transaction: {customer_made_transaction}")
        
        # Update the case in memory
        if customer_made_transaction:
            self.current_case["status"] = "confirmed_safe"
            self.current_case["outcome"] = f"Customer confirmed transaction as legitimate on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            response = f"Thank you for confirming. I've marked this transaction as safe in our system. "
            response += "Your card will remain active and no further action is needed. "
            response += "If you notice any other suspicious activity, please contact us immediately. Have a great day!"
        else:
            self.current_case["status"] = "confirmed_fraud"
            self.current_case["outcome"] = f"Customer denied transaction - fraud confirmed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            response = f"I understand. For your protection, I'm immediately blocking your card ending in {self.current_case['cardEnding']}. "
            response += "We will issue you a new card within 5 to 7 business days. "
            response += "We're also raising a dispute for this transaction, and the amount will be credited back to your account. "
            response += "You should receive an email confirmation shortly. Is there anything else I can help you with today?"
        
        # Save updated case back to database
        self._update_database()
        
        return response
    
    def _update_database(self):
        """Update the fraud cases database with current case status"""
        if not self.current_case:
            return
        
        # Find and update the case in the list
        for i, case in enumerate(self.fraud_cases):
            if case["userName"] == self.current_case["userName"]:
                self.fraud_cases[i] = self.current_case
                break
        
        # Save to file
        save_fraud_cases(self.fraud_cases)
        logger.info(f"Updated fraud case for {self.current_case['userName']}: {self.current_case['status']}")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # Preload fraud cases
    proc.userdata["fraud_cases"] = load_fraud_cases()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-priya",  # Professional Indian English voice
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
        agent=FraudAlertAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
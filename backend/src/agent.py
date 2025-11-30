print("I AM RUNNING THE NEW FIXED CODE")

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")


# -----------------------------------------------------------
# LOAD PRODUCT CATALOG
# -----------------------------------------------------------
def load_catalog():
    catalog_file = Path("shared-data/ecommerce_catalog.json")
    if catalog_file.exists():
        with open(catalog_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        logger.warning(f"Catalog not found at: {catalog_file}")
        return {"store_name": "TechStyle", "currency": "INR", "products": []}


CATALOG = load_catalog()


# -----------------------------------------------------------
# AGENT CLASS
# -----------------------------------------------------------
class EcommerceAgent(Agent):

    def __init__(self) -> None:
        store_name = CATALOG.get("store_name", "TechStyle India")

        super().__init__(
            instructions=f"""
You are a friendly shopping assistant for {store_name}.

Your job:
• Help customers browse items
• Search by color, category, or price
• Show product details
• Place orders
• Show last order

Always be natural, conversational, and helpful.
"""
        )

        self.recent_products = []
        self.last_order = None

    # -----------------------------------------------------------
    # SEARCH PRODUCTS (FULLY FIXED VERSION)
    # -----------------------------------------------------------
    @function_tool
    async def search_products(
        self,
        context: RunContext,
        category: Optional[str] = None,
        max_price: Optional[int] = None,
        color: Optional[str] = None,
        query: Optional[str] = None,
    ) -> str:

        logger.info(f"SEARCH: category={category}, max_price={max_price}, color={color}, query={query}")

        products = CATALOG.get("products", [])
        results = []

        # Make all fields optional without errors
        category = category or ""
        color = color or ""
        query = query or ""
        max_price = max_price or 0

        search_terms = []
        if category:
            search_terms.extend(category.lower().split())
        if query:
            search_terms.append(query.lower())

        for product in products:
            name = product.get("name", "").lower()
            desc = product.get("description", "").lower()
            cat = product.get("category", "").lower()
            prod_color = product.get("attributes", {}).get("color", "").lower()

            matched = True

            # Text searching
            if search_terms:
                if not any(term in name or term in desc or term in cat for term in search_terms):
                    matched = False

            # Filter by price
            if max_price > 0 and product.get("price", 0) > max_price:
                matched = False

            # Filter by color
            if color and color.lower() not in prod_color:
                matched = False

            if matched:
                results.append(product)

        self.recent_products = results[:5]

        if not results:
            return "I couldn't find any matching products."

        response = f"I found {len(results)} product(s):\n\n"
        for i, product in enumerate(results[:5], 1):
            response += f"{i}. {product['name']} - ₹{product['price']}"
            attr = product.get("attributes", {})
            if "color" in attr:
                response += f" ({attr['color']})"
            if "size" in attr:
                response += f" Size {attr['size']}"
            response += f"\n   {product['description']}\n"

        if len(results) > 5:
            response += f"\n...and {len(results) - 5} more."

        return response

    # -----------------------------------------------------------
    # PRODUCT DETAILS
    # -----------------------------------------------------------
    @function_tool
    async def get_product_details(self, context: RunContext, product_id: str) -> str:
        logger.info(f"DETAILS: product_id={product_id}")

        for product in CATALOG.get("products", []):
            if product.get("id") == product_id:
                msg = (
                    f"{product['name']} (₹{product['price']})\n"
                    f"{product['description']}\n\nDetails:\n"
                )
                for k, v in product.get("attributes", {}).items():
                    msg += f"- {k.title()}: {v}\n"

                msg += "\nIn Stock: Yes" if product.get("in_stock") else "\nIn Stock: No"
                return msg

        return f"Sorry, product with ID '{product_id}' not found."

    # -----------------------------------------------------------
    # CREATE ORDER
    # -----------------------------------------------------------
    @function_tool
    async def create_order(
        self,
        context: RunContext,
        product_ids: str,
        quantities: str = "1",
        buyer_name: str = "",
        buyer_email: str = "",
    ) -> str:

        logger.info(f"ORDER: product_ids={product_ids}, qty={quantities}")

        product_list = [p.strip() for p in product_ids.split(",")]
        qty_list = [int(q) for q in quantities.split(",")]

        if len(qty_list) < len(product_list):
            qty_list.extend([1] * (len(product_list) - len(qty_list)))

        line_items = []
        total = 0

        for pid, qty in zip(product_list, qty_list):
            product = next((p for p in CATALOG["products"] if p["id"] == pid), None)
            if not product:
                return f"Product not found: {pid}"

            price = product["price"]
            line_total = price * qty
            total += line_total

            line_items.append({
                "product_id": pid,
                "product_name": product["name"],
                "quantity": qty,
                "unit_amount": price,
                "line_total": line_total,
                "currency": "INR",
            })

        order = {
            "order_id": f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status": "CONFIRMED",
            "created_at": datetime.now().isoformat(),
            "buyer": {"name": buyer_name or "Guest", "email": buyer_email},
            "line_items": line_items,
            "total_amount": total,
            "currency": "INR",
        }

        Path("orders").mkdir(exist_ok=True)
        with open(f"orders/{order['order_id']}.json", "w", encoding="utf-8") as f:
            json.dump(order, f, indent=2, ensure_ascii=False)

        self.last_order = order

        msg = f"Order {order['order_id']} placed!\n\nItems:\n"
        for item in line_items:
            msg += f"- {item['product_name']} x{item['quantity']} = ₹{item['line_total']}\n"
        msg += f"\nTotal: ₹{total}\nThank you for shopping!"

        return msg

    # -----------------------------------------------------------
    # SHOW LAST ORDER
    # -----------------------------------------------------------
    @function_tool
    async def show_last_order(self, context: RunContext) -> str:
        if not self.last_order:
            return "No orders placed yet."

        order = self.last_order
        msg = f"Your last order ({order['order_id']}):\n\n"
        for item in order["line_items"]:
            msg += f"- {item['product_name']} x{item['quantity']} = ₹{item['line_total']}\n"
        msg += f"\nTotal: ₹{order['total_amount']}\nStatus: {order['status']}"

        return msg


# -----------------------------------------------------------
# WORKER SETUP
# -----------------------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["catalog"] = load_catalog()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-priya",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    async def summary():
        logger.info(f"Usage Summary: {usage.get_summary()}")

    ctx.add_shutdown_callback(summary)

    await session.start(
        agent=EcommerceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

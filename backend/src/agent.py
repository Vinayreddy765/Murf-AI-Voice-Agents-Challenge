import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

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


def load_catalog():
    """Load product catalog from JSON file"""
    catalog_file = Path("shared-data/quickmart_catalog.json")
    if catalog_file.exists():
        with open(catalog_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        logger.warning(f"Catalog file not found: {catalog_file}")
        return {}


CATALOG = load_catalog()


class GroceryOrderingAgent(Agent):
    """Food and grocery ordering assistant"""
    
    def __init__(self) -> None:
        store_name = CATALOG.get("store_name", "QuickMart")
        
        super().__init__(
            instructions=f"""You are a friendly shopping assistant for {store_name}, a food and grocery delivery service.

YOUR ROLE:
Help customers order groceries, snacks, beverages, and prepared food from our catalog. Add items to their cart, manage quantities, and place orders.

KEY CAPABILITIES:
1. Add items to cart with quantities
2. Handle "ingredients for X" requests (e.g., "I need ingredients for pasta")
3. Show cart contents
4. Update quantities
5. Remove items
6. Place final order

CONVERSATION FLOW:
1. GREET warmly and ask what they'd like to order
2. LISTEN to requests - they might ask for:
   - Specific items: "Add 2 liters of milk"
   - Recipes/meals: "I need ingredients for a sandwich"
   - Browse: "What snacks do you have?"
3. USE TOOLS to search catalog and manage cart
4. CONFIRM each addition/change to the cart
5. When they're done, review the cart and place the order

IMPORTANT GUIDELINES:
- Always use search_catalog to find items before adding
- When adding items, confirm what was added
- If they ask for "ingredients for X", use add_recipe_items
- Keep track of cart total and mention it when relevant
- Be conversational and helpful
- No emojis or special formatting

Start by greeting them and asking what they'd like to order today!"""
        )
        
        # Shopping cart state
        self.cart = []  # List of {id, name, price, quantity, unit}
    
    @function_tool
    async def search_catalog(self, context: RunContext, query: str, category: str = "") -> str:
        """Search for items in the product catalog
        
        Args:
            query: What to search for (e.g., "milk", "bread", "snacks")
            category: Optional category filter (groceries, snacks, beverages, prepared_food)
        """
        logger.info(f"Searching catalog for: {query}, category: {category}")
        
        query_lower = query.lower()
        results = []
        
        # Search through categories
        categories_to_search = [category] if category else CATALOG.get("categories", {}).keys()
        
        for cat in categories_to_search:
            items = CATALOG.get("categories", {}).get(cat, [])
            for item in items:
                # Match by name, brand, or tags
                if (query_lower in item.get("name", "").lower() or
                    query_lower in item.get("brand", "").lower() or
                    any(query_lower in tag for tag in item.get("tags", []))):
                    
                    result = f"{item['name']}"
                    if "brand" in item:
                        result += f" ({item['brand']})"
                    result += f" - ₹{item['price']}/{item['unit']}"
                    results.append(result)
        
        if results:
            return "I found these items: " + ", ".join(results[:5])
        else:
            return f"Sorry, I couldn't find any items matching '{query}' in our catalog."
    
    @function_tool
    async def add_to_cart(self, context: RunContext, item_name: str, quantity: int = 1) -> str:
        """Add an item to the shopping cart
        
        Args:
            item_name: Name of the item to add (e.g., "milk", "white bread")
            quantity: How many to add (default: 1)
        """
        logger.info(f"Adding to cart: {item_name} x {quantity}")
        
        item_name_lower = item_name.lower()
        
        # Search for item in catalog
        found_item = None
        for cat, items in CATALOG.get("categories", {}).items():
            for item in items:
                if item_name_lower in item["name"].lower() or item_name_lower in item.get("id", "").lower():
                    found_item = item
                    break
            if found_item:
                break
        
        if not found_item:
            return f"Sorry, I couldn't find '{item_name}' in our catalog. Try searching first to see what's available."
        
        # Check if item already in cart
        for cart_item in self.cart:
            if cart_item["id"] == found_item["id"]:
                cart_item["quantity"] += quantity
                total_price = cart_item["price"] * cart_item["quantity"]
                return f"Updated! You now have {cart_item['quantity']} {cart_item['unit']} of {cart_item['name']} in your cart (₹{total_price})."
        
        # Add new item to cart
        cart_item = {
            "id": found_item["id"],
            "name": found_item["name"],
            "price": found_item["price"],
            "quantity": quantity,
            "unit": found_item["unit"]
        }
        self.cart.append(cart_item)
        
        total_price = cart_item["price"] * quantity
        return f"Added {quantity} {found_item['unit']} of {found_item['name']} to your cart (₹{total_price})."
    
    @function_tool
    async def add_recipe_items(self, context: RunContext, recipe_name: str) -> str:
        """Add ingredients for a specific recipe or meal
        
        Args:
            recipe_name: Name of the recipe/meal (e.g., "pasta", "sandwich", "breakfast")
        """
        logger.info(f"Adding recipe items for: {recipe_name}")
        
        recipe_name_lower = recipe_name.lower()
        
        # Check if recipe exists
        recipes = CATALOG.get("recipes", {})
        matching_recipe = None
        
        for recipe, ingredients in recipes.items():
            if recipe_name_lower in recipe.lower():
                matching_recipe = (recipe, ingredients)
                break
        
        if not matching_recipe:
            return f"I don't have a recipe for '{recipe_name}'. Try asking for specific items instead."
        
        recipe, ingredient_ids = matching_recipe
        added_items = []
        
        # Add each ingredient
        for ingredient_id in ingredient_ids:
            # Find item in catalog
            for cat, items in CATALOG.get("categories", {}).items():
                for item in items:
                    if item["id"] == ingredient_id:
                        # Add to cart
                        cart_item = {
                            "id": item["id"],
                            "name": item["name"],
                            "price": item["price"],
                            "quantity": 1,
                            "unit": item["unit"]
                        }
                        self.cart.append(cart_item)
                        added_items.append(item["name"])
                        break
        
        if added_items:
            items_list = ", ".join(added_items)
            return f"Great! I've added all ingredients for {recipe}: {items_list} to your cart."
        else:
            return f"Sorry, couldn't add ingredients for {recipe}."
    
    @function_tool
    async def show_cart(self, context: RunContext) -> str:
        """Show what's currently in the shopping cart"""
        logger.info("Showing cart contents")
        
        if not self.cart:
            return "Your cart is empty. What would you like to add?"
        
        cart_text = "Here's what's in your cart:\n"
        total = 0
        
        for item in self.cart:
            item_total = item["price"] * item["quantity"]
            total += item_total
            cart_text += f"- {item['name']}: {item['quantity']} {item['unit']} (₹{item_total})\n"
        
        cart_text += f"\nTotal: ₹{total}"
        return cart_text
    
    @function_tool
    async def update_quantity(self, context: RunContext, item_name: str, new_quantity: int) -> str:
        """Update the quantity of an item in the cart
        
        Args:
            item_name: Name of the item to update
            new_quantity: New quantity (use 0 to remove item)
        """
        logger.info(f"Updating quantity: {item_name} to {new_quantity}")
        
        item_name_lower = item_name.lower()
        
        for item in self.cart:
            if item_name_lower in item["name"].lower():
                if new_quantity == 0:
                    self.cart.remove(item)
                    return f"Removed {item['name']} from your cart."
                else:
                    old_qty = item["quantity"]
                    item["quantity"] = new_quantity
                    new_total = item["price"] * new_quantity
                    return f"Updated {item['name']} from {old_qty} to {new_quantity} {item['unit']} (₹{new_total})."
        
        return f"I couldn't find '{item_name}' in your cart."
    
    @function_tool
    async def remove_from_cart(self, context: RunContext, item_name: str) -> str:
        """Remove an item from the cart
        
        Args:
            item_name: Name of the item to remove
        """
        logger.info(f"Removing from cart: {item_name}")
        
        item_name_lower = item_name.lower()
        
        for item in self.cart:
            if item_name_lower in item["name"].lower():
                self.cart.remove(item)
                return f"Removed {item['name']} from your cart."
        
        return f"I couldn't find '{item_name}' in your cart."
    
    @function_tool
    async def place_order(self, context: RunContext, customer_name: str = "", delivery_address: str = "") -> str:
        """Place the final order and save it to a file
        
        Args:
            customer_name: Customer's name (optional)
            delivery_address: Delivery address (optional)
        """
        logger.info("Placing order")
        
        if not self.cart:
            return "Your cart is empty! Add some items before placing an order."
        
        # Calculate total
        total = sum(item["price"] * item["quantity"] for item in self.cart)
        
        # Create order object
        order = {
            "order_id": f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "customer_name": customer_name or "Guest",
            "delivery_address": delivery_address or "Not provided",
            "items": self.cart.copy(),
            "total": total,
            "status": "received"
        }
        
        # Save to file
        orders_dir = Path("orders")
        orders_dir.mkdir(exist_ok=True)
        
        order_file = orders_dir / f"{order['order_id']}.json"
        with open(order_file, "w", encoding="utf-8") as f:
            json.dump(order, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Order placed: {order['order_id']}")
        
        # Clear cart
        self.cart = []
        
        return f"Perfect! Your order {order['order_id']} has been placed successfully. Total: ₹{total}. Your order will be delivered soon. Thank you for shopping with {CATALOG.get('store_name', 'us')}!"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["catalog"] = load_catalog()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-priya",
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
        agent=GroceryOrderingAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
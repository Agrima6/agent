"""Blue-collar job families: how a free-text role name (watchman, driver, electrician, ...) is recognised, what an
interviewer should probe for that kind of work, and a few role-specific bank questions per family.

Plain data plus small lookups. roles.py uses `ALL_KEYWORDS` to decide a role is blue collar; the planner uses
`family_for` / `focus_for` so a watchman and a warehouse loader are NOT asked the same questions; `FAMILY_QUESTIONS` is
merged into the question bank at load time (see planner.QUESTION_BANK).
"""

# family -> (label, keywords (lower case, English + common Hindi/Hinglish names), focus for question writing)
FAMILIES = {
    "security": (
        "Security / Watchman",
        ["security guard", "watchman", "chowkidar", "chaukidar", "gatekeeper", "gate keeper", "security supervisor", "bouncer",
         "night guard", "guard", "चौकीदार", "गार्ड"],
        "staying alert on long or night shifts, checking visitors and vehicles, logging entries, handling a suspicious "
        "person or an alarm, fire or emergency response, calm behaviour with angry visitors, reporting to the supervisor",
    ),
    "driving": (
        "Driver / Transport",
        ["driver", "chauffeur", "truck driver", "cab driver", "bus driver", "tempo", "school bus", "ड्राइवर", "चालक"],
        "road and vehicle safety, licence and documents, daily vehicle checks, route knowledge, handling a breakdown or "
        "accident, fatigue and night driving, politeness with passengers, fuel and trip records",
    ),
    "delivery": (
        "Delivery / Courier",
        ["delivery boy", "delivery partner", "delivery executive", "courier", "rider", "dispatch rider", "last mile",
         "डिलीवरी"],
        "on-time delivery, finding addresses, handling a customer who is not at home, cash on delivery, damaged or "
        "wrong parcels, safe riding, handling the app and proof of delivery",
    ),
    "warehouse": (
        "Warehouse / Logistics",
        ["warehouse", "godown", "loader", "unloader", "packer", "picker", "store keeper", "storekeeper",
         "forklift", "inventory helper", "logistics helper", "गोदाम", "पैकर"],
        "safe lifting and stacking, picking the right item, counting and labelling, damaged stock, forklift or "
        "trolley safety, working to a target, keeping the area clean",
    ),
    "housekeeping": (
        "Housekeeping / Cleaning",
        ["housekeeping", "house keeping", "cleaner", "janitor", "sweeper", "sanitation", "room attendant", "cleaning staff",
         "office boy", "peon", "ayah", "सफाई", "चपरासी"],
        "cleaning routines and checklists, chemicals and safe use, hygiene, attention to detail, handling guests' or "
        "clients' belongings honestly, reporting damage or lost items, working without supervision",
    ),
    "construction": (
        "Construction / Site",
        ["construction", "site worker", "mason", "mistri", "bricklayer", "painter", "carpenter", "plasterer", "scaffold",
         "shuttering", "bar bender", "tile fitter", "site helper", "labourer", "laborer", "mazdoor", "मिस्त्री", "मजदूर"],
        "site safety (helmet, harness, scaffolds), reading simple drawings or instructions, measuring, using tools "
        "properly, working at heights, teamwork with the gang and the supervisor, finishing quality",
    ),
    "electrical": (
        "Electrician / Electrical",
        ["electrician", "wireman", "lineman", "electrical technician", "ac technician", "hvac", "refrigeration", "इलेक्ट्रीशियन"],
        "electrical safety and isolation before work, fault finding step by step, using a multimeter and tools, "
        "wiring standards, earthing, working with a team, explaining a fault to a customer",
    ),
    "plumbing": (
        "Plumber / Pipe fitter",
        ["plumber", "pipe fitter", "pipefitter", "sanitary", "water pump", "borewell", "प्लंबर"],
        "finding leaks and blockages, fitting and testing, tools, water and hygiene safety, working in a customer's home "
        "tidily and politely, estimating materials",
    ),
    "mechanic": (
        "Mechanic / Technician",
        ["mechanic", "technician", "fitter", "service engineer", "garage", "two wheeler", "bike mechanic", "auto mechanic",
         "motor winding", "maintenance technician", "maintenance helper", "मैकेनिक", "टेक्नीशियन"],
        "diagnosing a fault, using tools and measuring instruments, preventive maintenance, spare parts, safety lock-out, "
        "documenting the repair, explaining the problem to a customer or supervisor",
    ),
    "welding": (
        "Welder / Fabricator",
        ["welder", "fabricator", "gas cutter", "grinder", "sheet metal", "turner", "machinist", "cnc operator", "lathe",
         "वेल्डर"],
        "welding or cutting safety (PPE, fire, fumes), reading drawings and measurements, joint quality and rework, "
        "machine setup, tool care, working to tolerance and deadlines",
    ),
    "factory": (
        "Factory / Machine operator",
        ["factory worker", "machine operator", "assembly line", "production helper", "production operator", "line worker",
         "operator", "packing operator", "quality checker", "textile operator", "ऑपरेटर", "फैक्ट्री"],
        "machine safety, following the work instruction, quality checks, reporting a defect or an abnormal sound, "
        "shift handover, targets, teamwork on the line, housekeeping (5S)",
    ),
    "kitchen": (
        "Kitchen / Hospitality",
        ["cook", "chef", "kitchen staff", "kitchen helper", "commis", "dishwasher", "waiter", "steward", "bartender",
         "barista", "canteen", "tandoor", "baker", "रसोइया", "वेटर"],
        "food hygiene and safety, preparing to a recipe or order, timing in a rush, handling a customer complaint, "
        "cleanliness, kitchen safety (knives, gas, oil), teamwork",
    ),
    "retail": (
        "Retail / Shop floor",
        ["salesman", "sales boy", "shop assistant", "counter staff", "store helper", "cashier", "showroom helper",
         "shop helper"],
        "greeting and helping customers, product knowledge, handling cash honestly, a customer who is angry or wants "
        "a refund, stock on shelves, working in a rush, targets",
    ),
    "tailoring": (
        "Tailor / Textile",
        ["tailor", "seamstress", "stitching", "embroidery", "textile worker", "weaver", "darzi", "दर्जी"],
        "measurements, cutting and stitching accuracy, machine care, deadlines before festivals, fixing a mistake, "
        "customer fittings and complaints",
    ),
    "agriculture": (
        "Agriculture / Field",
        ["farm worker", "farmhand", "farm hand", "gardener", "mali", "field staff", "plantation", "tractor",
         "dairy", "poultry", "किसान", "माली"],
        "daily field routine, tools and machines, pesticide and water safety, weather problems, animals' or plants' care, "
        "reporting a problem early, working long hours",
    ),
    "domestic": (
        "Domestic / Care",
        ["maid", "domestic help", "nanny", "caretaker", "care taker", "patient care", "nursing aide",
         "ward boy", "home help", "बाई", "केयरटेकर"],
        "honesty and trust in a client's home, routines, hygiene, caring for children or elderly people, emergencies, "
        "respect for privacy, following the family's instructions",
    ),
    "general": (
        "General helper / Field worker",
        ["helper", "field worker", "general worker", "utility worker", "hamal", "हेल्पर"],
        "reliability and punctuality, following instructions, simple tools, teamwork, safety, willingness to learn on the job",
    ),
}

# Order matters: the first family whose keyword appears in the role name wins (specific before generic).
_FAMILY_ORDER = ["security", "delivery", "driving", "warehouse", "electrical", "plumbing", "welding", "construction",
                 "mechanic", "kitchen", "housekeeping", "tailoring", "agriculture", "domestic", "retail", "factory", "general"]

ALL_KEYWORDS = sorted({kw for _, kws, _ in FAMILIES.values() for kw in kws}, key=len, reverse=True)


def family_for(role_name: str) -> str | None:
    name = f" {(role_name or '').lower()} "
    for key in _FAMILY_ORDER:
        if any(kw in name for kw in FAMILIES[key][1]):
            return key
    return None


def focus_for(role_name: str) -> str:
    family = family_for(role_name)
    return FAMILIES[family][2] if family else ""


def label_for(role_name: str) -> str:
    family = family_for(role_name)
    return FAMILIES[family][0] if family else ""


def _q(family: str, n: int, text: str, comps: list[str], topics: list[str], follow: list[str]) -> dict:
    return {"id": f"q_bc_{family}_{n}", "type": "scenario", "difficulty": "easy", "role_tags": ["blue_collar", f"bc_{family}"],
            "competencies": comps, "question_text": text, "expected_topics": topics, "followup_topics": follow}


FAMILY_QUESTIONS = [
    _q("security", 1, "It is two in the night and you see someone you do not know walking towards the gate. What will you do?",
       ["safety_awareness", "reliability"], ["stop and ask", "verify", "inform supervisor", "log entry"], ["if he refuses", "if he says he is a guest"]),
    _q("security", 2, "A visitor is angry because you are not letting him in without a pass. How do you handle it?",
       ["following_instructions", "reliability"], ["stay calm", "explain rule", "call the person", "escalate"], ["if he shouts", "if he is a senior person"]),
    _q("security", 3, "You are on a long night shift and feel very sleepy. What do you do to stay alert and keep the place safe?",
       ["reliability", "safety_awareness"], ["walk rounds", "water", "stay in contact", "never leave post"], ["how often do you patrol", "what do you note down"]),
    _q("driving", 1, "Before you start the vehicle in the morning, what all do you check?",
       ["safety_awareness", "practical_skill"], ["tyres", "brakes", "oil and fuel", "lights", "documents"], ["what if you find a problem", "who do you tell"]),
    _q("driving", 2, "The vehicle breaks down on the highway with a load or passengers inside. What will you do first?",
       ["safety_awareness", "practical_skill"], ["park safely", "indicators", "inform owner", "stay with vehicle"], ["at night", "if it is raining"]),
    _q("driving", 3, "You are running late and the road ahead is jammed. Someone tells you to take a risky shortcut. What do you decide?",
       ["following_instructions", "reliability"], ["safety first", "inform customer", "legal route"], ["if the owner insists", "if you lose the trip"]),
    _q("delivery", 1, "The customer is not picking up the phone and you are at the address. What do you do?",
       ["reliability", "practical_skill"], ["call again", "ask neighbours", "contact support", "wait a little"], ["how long do you wait", "what do you do with the parcel"]),
    _q("delivery", 2, "You find the parcel is damaged before you reach the customer. What will you do?",
       ["reliability", "following_instructions"], ["inform support", "do not hide", "photo", "return process"], ["if the customer still wants it", "if the app does not allow return"]),
    _q("delivery", 3, "You have ten deliveries and not enough time. How do you plan your route?",
       ["practical_skill", "reliability"], ["nearby addresses", "priority orders", "traffic", "inform delay"], ["what if one order is far away", "what if the bike fails"]),
    _q("warehouse", 1, "How do you lift and carry a heavy box safely?",
       ["safety_awareness", "practical_skill"], ["bend knees", "straight back", "ask for help", "use trolley"], ["what if no one helps", "what if the box is slippery"]),
    _q("warehouse", 2, "You are asked to pick ten items but you find only eight on the shelf. What do you do?",
       ["reliability", "following_instructions"], ["check nearby", "tell supervisor", "do not guess", "record shortage"], ["if the supervisor is not there", "if the count is not matching"]),
    _q("warehouse", 3, "You see a wet floor and a pile of boxes leaning near the walkway. What will you do?",
       ["safety_awareness", "teamwork"], ["warn others", "fix it or report", "mark area", "tell supervisor"], ["if you are in a hurry", "if someone else left it there"]),
    _q("housekeeping", 1, "How will you clean a washroom so that it is ready for the next guest?",
       ["practical_skill", "reliability"], ["gloves", "chemical use", "order of work", "dry and check"], ["which chemicals", "what if there is a bad smell"]),
    _q("housekeeping", 2, "While cleaning a room you find a wallet on the table. What will you do?",
       ["reliability", "following_instructions"], ["do not touch much", "inform supervisor", "hand over", "honesty"], ["if nobody is around", "if it has a lot of cash"]),
    _q("housekeeping", 3, "You are told to mix two cleaning chemicals to make them stronger. What will you do?",
       ["safety_awareness", "following_instructions"], ["never mix", "read label", "ask supervisor", "ventilation"], ["what happens if mixed", "what if the supervisor says do it"]),
    _q("construction", 1, "You have to work on the second floor outside the building. What safety things do you make sure of?",
       ["safety_awareness", "following_instructions"], ["harness", "helmet", "scaffold check", "do not rush"], ["if there is no harness", "if it starts raining"]),
    _q("construction", 2, "The supervisor gives you a measurement but you think it is wrong. What do you do?",
       ["following_instructions", "reliability"], ["measure again", "ask politely", "do not just proceed", "tell supervisor"], ["if he insists", "if you are not sure"]),
    _q("construction", 3, "A co-worker is working without a helmet near falling materials. What will you do?",
       ["safety_awareness", "teamwork"], ["tell him", "give helmet", "report if repeated", "stop work if needed"], ["if he is senior", "if he gets angry"]),
    _q("electrical", 1, "Before you touch a switchboard or wiring for repair, what do you do first?",
       ["safety_awareness", "practical_skill"], ["switch off mains", "tester", "lock or tag", "insulated tools"], ["what if someone switches it on", "what do you do if the tester shows current"]),
    _q("electrical", 2, "A customer says the lights keep tripping the MCB. How will you find the problem?",
       ["practical_skill", "reliability"], ["isolate circuits", "check load", "loose connection", "insulation"], ["what tools do you use", "what if you cannot find the fault"]),
    _q("electrical", 3, "Someone gets an electric shock at the site. What will you do immediately?",
       ["safety_awareness", "reliability"], ["switch off power", "do not touch directly", "dry insulator", "call for help"], ["if the person is not breathing", "who do you inform"]),
    _q("plumbing", 1, "A customer has a leak under the kitchen sink. How will you find and fix it?",
       ["practical_skill", "reliability"], ["close supply", "find joint", "replace washer or fitting", "test"], ["what tools", "what if the pipe is old"]),
    _q("plumbing", 2, "You finish the work but the customer's floor is wet and dirty. What do you do?",
       ["reliability", "teamwork"], ["clean", "dry floor", "respect home", "confirm with customer"], ["what if the customer is not at home", "what if you are late for the next job"]),
    _q("plumbing", 3, "The customer wants a cheaper fitting that you know will not last. What will you tell them?",
       ["reliability", "following_instructions"], ["be honest", "explain risk", "give options", "their choice"], ["if they refuse", "if the owner wants it done anyway"]),
    _q("mechanic", 1, "A machine or vehicle is making a strange noise. How will you find out what is wrong?",
       ["practical_skill", "safety_awareness"], ["listen and locate", "switch off", "check basic things", "step by step"], ["what tools", "what if you are not sure"]),
    _q("mechanic", 2, "You open a unit for repair and find another problem that was not told to you. What do you do?",
       ["reliability", "following_instructions"], ["tell customer or supervisor", "do not hide", "estimate", "get approval"], ["if the customer refuses", "if it is urgent"]),
    _q("mechanic", 3, "You are repairing something and a spare part is not available. What will you do?",
       ["practical_skill", "reliability"], ["inform", "alternate plan", "do not use wrong part", "time estimate"], ["if the customer is in a hurry", "if a local shop has a similar one"]),
    _q("welding", 1, "What safety things do you use and check before you start welding?",
       ["safety_awareness", "practical_skill"], ["helmet and gloves", "no fuel nearby", "ventilation", "earthing"], ["what if there is a fire", "what about other workers nearby"]),
    _q("welding", 2, "After welding, the joint looks weak or has gaps. What do you do?",
       ["practical_skill", "reliability"], ["check and redo", "inform supervisor", "do not pass bad work"], ["how do you test it", "what if the deadline is close"]),
    _q("welding", 3, "You are given a drawing with measurements. How do you make sure the piece comes out right?",
       ["following_instructions", "practical_skill"], ["read carefully", "measure twice", "mark before cutting", "ask if unclear"], ["what if the drawing is unclear", "what if material is short"]),
    _q("factory", 1, "While operating a machine you notice it is vibrating more than usual. What will you do?",
       ["safety_awareness", "reliability"], ["stop machine", "tell supervisor", "do not repair yourself", "tag it"], ["if the target is pending", "if a colleague says it is fine"]),
    _q("factory", 2, "You find that some of the items you made are not matching the sample. What will you do?",
       ["reliability", "following_instructions"], ["stop and check", "inform supervisor", "keep separate", "do not pass it"], ["if the line is waiting", "if only you noticed it"]),
    _q("factory", 3, "Your shift is ending and the next person has not come. How will you hand over the work?",
       ["teamwork", "reliability"], ["inform supervisor", "write handover", "explain status", "do not leave machine unattended"], ["if the supervisor is not there", "if you are very tired"]),
    _q("kitchen", 1, "The kitchen is very busy and many orders come together. How will you manage?",
       ["practical_skill", "teamwork"], ["priority", "communication", "clean as you go", "stay calm"], ["what if one order is wrong", "what if a team member is absent"]),
    _q("kitchen", 2, "You see that some vegetables or food have gone bad. What will you do?",
       ["safety_awareness", "reliability"], ["throw away", "inform chef", "never serve", "check others"], ["what if the chef says use it", "what if stock is short"]),
    _q("kitchen", 3, "A guest says the food is cold and is upset. How do you respond?",
       ["reliability", "teamwork"], ["listen", "apologise", "inform chef", "replace quickly"], ["if the guest is rude", "if the item is finished"]),
    _q("retail", 1, "A customer wants to return an item and the bill is not with them. What do you do?",
       ["following_instructions", "reliability"], ["polite", "check policy", "call manager", "do not promise"], ["if the customer argues", "if the manager is not available"]),
    _q("retail", 2, "You gave extra change to a customer by mistake after they left. What would you do?",
       ["reliability", "following_instructions"], ["inform manager", "be honest", "record it", "do not hide"], ["if it is small amount", "if the customer returns"]),
    _q("retail", 3, "A customer asks about a product you do not know well. How do you help?",
       ["practical_skill", "teamwork"], ["be honest", "ask colleague", "check tag", "come back with answer"], ["if the customer is in a hurry", "if no one else knows"]),
    _q("tailoring", 1, "A customer wants the dress by tomorrow but you have many orders. What will you do?",
       ["reliability", "practical_skill"], ["be honest about time", "prioritise", "ask for help", "do not rush quality"], ["if they insist", "if the cloth is difficult"]),
    _q("tailoring", 2, "After stitching you see that the measurement is slightly wrong. What do you do?",
       ["reliability", "practical_skill"], ["tell customer or owner", "fix it", "do not hide", "check measurement again"], ["if cloth is already cut", "if they have already paid"]),
    _q("tailoring", 3, "How do you take correct measurements for a shirt or blouse?",
       ["practical_skill", "following_instructions"], ["tape", "chest waist length", "ask preference", "note down"], ["how do you handle a person who keeps moving", "what about fitting"]),
    _q("agriculture", 1, "You find that some plants or animals in your care look sick. What do you do?",
       ["reliability", "practical_skill"], ["observe", "separate", "inform owner", "do not delay"], ["what do you check first", "what if the owner is away"]),
    _q("agriculture", 2, "How do you safely use pesticide or any chemical in the field?",
       ["safety_awareness", "following_instructions"], ["gloves and mask", "right quantity", "wind direction", "wash after"], ["what if you feel dizzy", "where do you store it"]),
    _q("agriculture", 3, "A sudden rain starts while the work is half done. What do you do?",
       ["practical_skill", "reliability"], ["protect crop and tools", "inform", "safe shelter", "resume later"], ["what if the owner wants to continue", "what if equipment is out"]),
    _q("domestic", 1, "The family's child or an elderly person is not feeling well when you are alone with them. What will you do?",
       ["reliability", "safety_awareness"], ["stay calm", "call family", "emergency number", "do not give medicine on own"], ["if no one answers", "if it is night"]),
    _q("domestic", 2, "You accidentally break something valuable in the house. What will you do?",
       ["reliability", "following_instructions"], ["tell immediately", "be honest", "do not hide", "apologise"], ["if nobody saw", "if you are afraid of losing the job"]),
    _q("domestic", 3, "The family gives you instructions that are different from the previous family you worked with. How do you handle it?",
       ["following_instructions", "teamwork"], ["listen carefully", "ask questions", "follow their way", "note down"], ["if you think the old way was better", "if you forget"]),
    _q("general", 1, "You are told to do a task you have never done before. What will you do?",
       ["following_instructions", "practical_skill"], ["ask", "watch carefully", "start slowly", "do not guess on safety"], ["if no one is available to show", "if the supervisor is angry"]),
    _q("general", 2, "You reach work and see that two people from your team are absent. How will you manage the work?",
       ["teamwork", "reliability"], ["inform supervisor", "prioritise", "help each other", "do not leave work"], ["if you get extra work", "if the supervisor is not around"]),
]

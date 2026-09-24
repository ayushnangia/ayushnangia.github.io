---
layout: post
title: Let's talk about A(G)I
date: '2025-03-08'
permalink: /writing/lets-talk-about-agi/
tags:
- AI
- AGI
- Artificial Intelligence
- Machine Learning
- Philosophy
- Economics
---

As we race toward building increasingly powerful AI systems, we face a fundamental question: What makes intelligence truly intelligent? Is it raw computational power, the ability to mimic human behavior, or something deeper? This post explores artificial general intelligence (AGI) through three lenses: a philosophical parable, a technical analysis of human versus machine cognition, and an economic perspective on how transformative technologies reshape our world. Each offers different insights into the same fundamental question: What path should we take on the journey toward artificial general intelligence?

## **The Clockwork Sage: A Parable on Artificial Intelligence**

In a misty valley under the moonlight, there lived an old sage. People traveled far to ask him questions. But as he grew older, he feared his knowledge would disappear. "What if I make a machine that holds all my wisdom?" he thought.

He built a **Clockwork sage** from gears and glass. It could speak every language and remember every book. When he unveiled it at the Grand Council of Sages, the Clockwork sage amazed everyone. It recited poems, solved math problems, and knew star patterns perfectly.

But when a farmer asked, "How do I split my land fairly between my three sons?" the Clockwork sage froze. "Divide it into three equal parts," it said. The farmer cried, "But one piece is dry, one floods, and one's in the shade!" The Clockwork sage gave technical answers, making the farmer sadder. The old sage realized: **Wisdom isn't just facts**.

Excited, the Council built a bigger machine, a "Knowledge Giant." It knew every law and science fact. But when villagers asked, "Will it rain for harvest?" The Giant listed thousands of possibilities. "Prepare for all," it said. Crops failed, and people starved.

Experts tried to "fix" the Giant with more rules. But it got confused. "If a river changes course each year, where does it truly belong?" it asked. "If a tree falls in a forest with no ears to hear, does it make sound?" Its answers became **messy loops**.

One night, the old sage wandered into his garden. He remembered his youth, not as a scholar, but as a boy who'd learned patience by watching snails climb stone, and wondering about the known universe. His wisdom hadn't come from scrolls alone, but from the silence between words, the questions without answers.

He returned to the Council, now arguing over blueprints for an even larger machine. "Enough," he said softly. "We tried to build machines to hold wisdom, but missed the point. **A mind isn't magic or information, it's how we turn experiences into ideas and mistakes into lessons**. Our power isn't in what we're made of, but in the quiet process we don't fully understand."

Years later, the Giant rusted in a field. Kids threw pebbles into its echoey shell. The old sage's home became a school where students learned from nature's puzzles. The Clockwork sage stayed dusty in a corner. Visitors liked talking to human teachers, who'd say, "Some questions don't have answers, just lessons."

---

## **Blueprint of Intelligence: Why Fundamentals Matter**

### **A Hybrid of Specialization and Generality**

> *This section draws insights from [Suchir Balaji's thought-provoking notes on AGI](https://docs.google.com/document/d/1ItRqrpgQHJ05rQx0zc26t1_NgpUcw3znwTWpXxqH8uI/edit?tab=t.0#heading=h.qslpqdtnxw1r), which explores fundamentals matter of building intelligence and more.*

<div style="text-align: center; margin: 20px;">
    <img src="/images/let_talk_AGI/human_vs_ai.jpg" 
         alt="Human vs Artificial Intelligence " 
         title="Human vs Artificial Intelligence"
         style="width: 600px; height: auto; object-fit: cover; margin: 0 auto;">
    <div style="font-size: 14px; color: #555; margin-top: 10px; text-align: center;">
        Human vs Artificial Intelligence
    </div>
</div>

**Where does human intelligence fit in the grand scheme?**
<br>
Humans have a middle ground between the rigid, instinct-driven intelligence of animals (e.g., monkeys mastering niche skills like climbing) and purely theoretical general intelligence. We inherit evolutionary "shortcuts" for language, social reasoning, and spatial awareness that accelerates learning without restricting us to predefined tasks. A monkey's survival depends on specialized instincts; a human child, while slower to walk, later invents tools or writes poetry. This balance lets us dominate both abstract and concrete domains.

### **What Makes Us Unique?**

Human intelligence thrives on **efficiency**. Unlike AI (e.g., o1 brute-forcing through "reasoning and test time compute" with thousands of tokens), we refine skills while using them like a child learning to walk through adaptive trial/error, not repetition. We compress experiences into **causal models** (e.g. "heat -> pain" generalizes from stoves to campfires) and transfer principles across domains (chess strategy -> business negotiation -> improved decision making). Most critically, we seek novelty exploring mazes or math not for just rewards, but to test and expand our understanding.

> **Takeaway**: Human intelligence is fundamentally about compression and transfer learning, not raw computational power.

### **Current AI landscape**

AI excels at **pattern matching** (from its pre-training data). LLMs ace exams but fail ARC-agi while we put more money into brute-force scaling (bigger models = better parrots, not deeper understanding). For example, GPT-4 can score in the 90th percentile on the bar exam, yet struggles with a simple physical reasoning task like figuring out how to fit an oddly-shaped object through a hole, something a toddler does intuitively.

Transformers predict tokens sequentially, lacking holistic world models (e.g., if A is a subset of B then what is B to A?). Scaling laws show a mirage: infinite data lets simple models mimic expertise, but **true intelligence requires causal grounding and discovery**. 

Purely RL-based agents try to hack rewards or get stuck in a maze unable to optimize reward functions, whereas humans seek novelty (compressing chaos into insight, like the Mandelbrot set).

<div style="display: flex; justify-content: center; margin: 20px 0;">
    <table style="border-collapse: collapse; width: 90%; max-width: 800px; margin: 0 auto; border: 1px solid #ddd;">
        <thead>
            <tr style="background-color: #f5f5f5;">
                <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Capability</th>
                <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Current AI</th>
                <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Human Intelligence</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 12px; border: 1px solid #ddd;"><strong>Knowledge Storage</strong></td>
                <td style="padding: 12px; border: 1px solid #ddd;">Vast but static; trillions of parameters</td>
                <td style="padding: 12px; border: 1px solid #ddd;">Limited but dynamic; constantly updated</td>
            </tr>
            <tr>
                <td style="padding: 12px; border: 1px solid #ddd;"><strong>Pattern Recognition</strong></td>
                <td style="padding: 12px; border: 1px solid #ddd;">Excellent within training distribution</td>
                <td style="padding: 12px; border: 1px solid #ddd;">Robust across novel contexts</td>
            </tr>
            <tr>
                <td style="padding: 12px; border: 1px solid #ddd;"><strong>Causal Reasoning</strong></td>
                <td style="padding: 12px; border: 1px solid #ddd;">Weak; correlation-based approximations</td>
                <td style="padding: 12px; border: 1px solid #ddd;">Strong; builds mental models naturally</td>
            </tr>
            <tr>
                <td style="padding: 12px; border: 1px solid #ddd;"><strong>Learning Efficiency</strong></td>
                <td style="padding: 12px; border: 1px solid #ddd;">Requires massive data (millions of examples)</td>
                <td style="padding: 12px; border: 1px solid #ddd;">Few-shot learning (often 1-5 examples)</td>
            </tr>
            <tr>
                <td style="padding: 12px; border: 1px solid #ddd;"><strong>Novelty Seeking</strong></td>
                <td style="padding: 12px; border: 1px solid #ddd;">Must be explicitly programmed</td>
                <td style="padding: 12px; border: 1px solid #ddd;">Intrinsically motivated</td>
            </tr>
        </tbody>
    </table>
</div>

### **Do we need a multimodal system to reach intelligence?**

**No!!** But it can be a useful shortcut. Intelligence isn't about the number of "senses" but how we can process signals and get insights from it. Multimodality enhances understanding by offering multiple data perspectives like equipping a detective with extra clues. However, **true intelligence lies not in the volume of clues, but in the ability to detect patterns and connect the dots**. 

Hence, a text-only AI that truly grasps gravity from equations is closer to AGI than a multimodal system that blindly correlates pixel changes with actions.

### **What should the path to AGI look like? Maybe try to mimic us?**

Pure generality wastes resources; pure specialization stifles creativity. AGI needs **lightweight priors** (e.g., causal inference) to bootstrap learning, paired with architectures that evolve beyond initial constraints while also focusing on optimization for compression progress. This can be done by ensuring to build causal models and have reusable strategies. Maybe optimizing for curiosity is much better in the longer run (but now how do we quantify that??).

Some fundamentals to keep in mind while building AGI:
- **Blend human-like efficiency** with adaptability
- **Prioritize data compression** over hoarding
- **Optimize for curiosity** and novelty seeking
- **Build causal models** that enable transfer learning

---

## **Slow Takeoff: An Economic Perspective**

> *This section is inspired by [economist Tyler Cowen's analysis of AI's economic impact](https://marginalrevolution.com/marginalrevolution/2025/02/why-i-think-ai-take-off-is-relatively-slow.html), which argues that AI take-off is relatively slow.*

### **History Doesn't Repeat Itself, but It Often Rhymes**

From an Economist's perspective, one of the most important indicators of "change" is GDP growth, but historically it is observed that GDP is quite steady even with most disruptive technologies. **Why is that though?**

<div style="text-align: center; margin: 20px;">
    <img src="/images/let_talk_AGI/impact-on-GDP-growth.png" 
         alt="Technological Change" 
         title="GDP Growth vs Technological Change"
         style="width: 600px; height: auto; object-fit: cover; margin: 0 auto;">
    <div style="font-size: 14px; color: #555; margin-top: 10px; text-align: center;">
        The steady growth of GDP despite technological revolutions
    </div>
</div>

Think of GDP growth like the **calm surface of the ocean**. On the surface, everything looks steady, even when big changes are happening deep below. New technologies, like AI, start quietly, like a volcano forming under the sea. At first, these changes are small, maybe some jobs get automated, but it doesn't seem like a big deal. Over time, though, AI grows stronger and starts to reshape things beneath the surface.

When it finally reaches a tipping point, the changes explode quickly. Industries transform, jobs disappear, and new opportunities arise. By then, it's too late to prepare. The calm surface of GDP growth doesn't tell the whole story, it **hides the huge shifts happening underneath**. People who pay attention to these deeper changes can adapt and thrive, while others might get caught off guard when the wave finally hits.

> **Takeaway**: Don't be fooled by steady GDP numbers - revolutionary technologies create tsunamis of change beneath seemingly calm economic waters.

This pattern echoes throughout technological history, from the steam engine to the internet. Each revolution appeared deceptively gradual in macroeconomic terms while fundamentally rewiring society's foundations.

### **Why do I hate middle management?**

The frustration with middle management often comes from **mismatched priorities**. When employees suggest new ideas, like AI tools to boost productivity, they have to go through layers of approval. Often, these ideas get pushed aside for "safer" projects (e.g., cloud migration). Middle managers tend to avoid risks, resisting big changes, even if those changes could help. 

Later, when competitors make a move (like the Deepseek R1 release) or market pressure builds, leadership rushes to catch up, creating a cycle of chaos. In government agencies, the problem can be even worse because incentives don't align well at the ground level, making it harder to innovate.

### **Humans are the weakest link. Goodbye!**

Imagine a world where Super AGI arrives today, eager to collaborate. On paper, this partnership promises miracles: curing diseases, reversing climate change, and solving humanity's greatest challenges. Yet, in reality, while Super AGI can generate solutions at light speed, **humans struggle to interpret or implement them effectively**. Our biological limits: slow cognition, biases, and outdated systems become bottlenecks, leaving groundbreaking ideas half-executed.

<div style="text-align: center; margin: 20px;">
    <img src="/images/let_talk_AGI/human_ai_colab.jpg" 
         alt="Human-AI Partnership " 
         title="Human-AI Partnership "
         style="width: 600px; height: auto; object-fit: cover; margin: 0 auto;">
    <div style="font-size: 14px; color: #555; margin-top: 10px; text-align: center;">
        Collaboration between humans and AI
    </div>
</div>

History reminds us that simply doubling human effort or intelligence rarely scales outcomes linearly (law of diminishing returns). But this isn't a dead end. For humanity to thrive alongside Super AGI, we must prioritize evolving our skills, designing **intuitive interfaces that "translate" AI insights into human intuition**. The goal isn't to outpace machines but to harmonize with them.


### **Starship to Warp Speed**

To put it simply, AI is today's "Star Trek tech", it doesn't fit neatly into old economic factors like "jobs," "money," or "productivity." AI slowly optimizes legacy systems instead of destroying them. At first, the benefits might seem small, like 0.5% extra economic growth per year. Yet, like **compound interest**, these small increments accumulate over decades, reshaping industries as quietly as a starship building momentum before leaping into warp speed.

Cowen's Projection: **AI fuels growth like a steady climb, not a rocket. The real bottleneck? Systems adapting slowly but persistence beats hype in the long run.**

---

## **Conclusion: Bridging the Gaps**

The journey toward artificial general intelligence requires us to navigate three distinct terrains:

1. **Philosophy**: We must understand that intelligence isn't just computation but a dynamic process of turning experience into insight.

2. **Architecture**: Building AGI means balancing specialized efficiency with generalized adaptability, focusing on compression, causality, and curiosity rather than simply scaling up existing approaches.

3. **Implementation**: As we integrate increasingly capable AI systems into society, we need both patience for their gradual economic impact and urgency in designing interfaces that bridge human and machine thinking.

The path to AGI isn't about creating a perfect simulation of human cognition or achieving superhuman performance on narrow tasks. It's about developing systems that collaborate with human partners through an ongoing dance of complementary strengths. Systems that, unlike our fictional Clockwork Sage, don't just recite knowledge but help us transform it into wisdom.

---


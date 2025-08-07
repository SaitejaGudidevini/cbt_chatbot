import requests
import json

def format_conversation(raw_conversation):
    """Format a raw conversation into the expected format for the CBT evaluator."""
    # Split the conversation into lines
    lines = raw_conversation.strip().split('\n')
    
    formatted_text = ""
    current_speaker = None
    current_message = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a new speaker
        if line.startswith('>>>'):
            # User message
            if current_speaker:
                # Add the previous message
                formatted_text += f"{current_speaker}: {current_message.strip()}\n\n"
            
            current_speaker = "User"
            current_message = line.replace('>>>', '').strip()
        elif current_speaker == "User" and not line.startswith('>>>'):
            # This is the assistant's response
            formatted_text += f"User: {current_message.strip()}\n\n"
            current_speaker = "Assistant"
            current_message = line
        elif current_speaker == "Assistant":
            # Continue the assistant's message
            current_message += " " + line
    
    # Add the last message
    if current_speaker:
        formatted_text += f"{current_speaker}: {current_message.strip()}\n\n"
    
    return formatted_text.strip()

def evaluate_conversation(conversation_text, api_url="http://127.0.0.1:5009/evaluate"):
    """Send a conversation to the CBT evaluator API and get the scores."""
    response = requests.post(
        api_url,
        headers={"Content-Type": "application/json"},
        json={"conversation": conversation_text}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Example conversation
raw_conversation = """
Hello! I'm here to chat with you about anything on your mind. How's your day been so far?

>>> Hey there!
Hi! It's great to hear from you. What's been going on lately? Anything interesting happening in your world?

>>> nothing. I am alright just having lot of things running in my mind
I see. Sometimes it can feel overwhelming when we have a lot of thoughts buzzing around. Is there something specific that's been weighing on your mind lately? You don't have to go into details if you 
don't want to, but sometimes talking about it can help sort through those thoughts.

>>> sure, I feel like I am working alot but not productive. I am 25 and I am working for almost 16hours a day thinking about the future all the time. it's just confusing like my age people are having fun, 
... having girlfriends, here I am working extensively.
Thank you for sharing that with me. It sounds like a tough situation. Could you tell me what moods you're feeling right now, and maybe rate their intensity from 0 to 100%?

>>> 75 maybe
Got it. So, you're feeling around 75% of something challenging right now. What evidence from your situation supports the view that "I am working a lot but not productive"? Are there specific moments 
or tasks where you felt this way?

>>> I want to want hard and do multiple things, but when I see the end of the day, every work is in pending. that 's where I feel like I am working hard but not productive.
Okay, thank you for sharing that perspective. Now, can you think of any experiences, even small ones, that might contradict the view that "I'm working a lot but not productive"? For example, are 
there any tasks or projects where you felt a sense of accomplishment or progress?

>>> yea, I have been working on a AI project which is still pending, besides, my boss gave me a task that has to be ocmpleted soon, but still I am working on that project. 
That's a good example. Can you think of any others? Sometimes even little things count.

>>> right not that's all
Alright, thank you for sharing those examples. Now that we've looked at both perspectives, how could you rephrase your original thought in a way that feels more balanced to you, considering the 
evidence and experiences you mentioned?

>>> what do you mean?
I mean, given what you've told me about working on the AI project and completing tasks for your boss, can you find a way to describe your situation that acknowledges both the challenges and the 
progress you're making? For example, instead of saying "I'm working hard but not productive," how might you phrase it in a more balanced way?

>>> I am trying to work and complete it but I am failing to complete the task on time.
Now that we've looked at examples for and against your original thought, how are you feeling now? Earlier you mentioned feeling around 75%. How would you rate those same moods on a scale of 0-100% 
now?

>>> not at all. still 75%. in fact, I have to work now but I am chatting with you.
I understand that you feel like you're not completing tasks on time and that's contributing to your current mood. Let's try to find a more balanced way of looking at your situation.

You mentioned working hard on an AI project and having another task from your boss. These are significant responsibilities. It's possible to acknowledge the challenges while also recognizing your 
efforts and progress.

Could you maybe say something like, "I'm trying to balance multiple important tasks, which is challenging, but I am making progress even if it's not as fast as I'd like"?

>>> what happens if I say that?
If you rephrase your thought in a more balanced way, such as "I'm trying to balance multiple important tasks, which is challenging, but I am making progress even if it's not as fast as I'd like," you 
might start to see your situation from a different perspective. This can help reduce some of the negative feelings and give you a clearer view of what you're actually achieving.

How does that sound to you?

>>> it is great, but I want a practical solution. I want a solution that just helps my mind to get back to work and get my things together.
I understand that you're looking for a practical solution to help you refocus on your tasks. Here's what we can do:

First, let's try setting small, manageable goals for yourself. Break down your tasks into smaller steps that you can complete in shorter periods of time. This will make them feel less overwhelming 
and more achievable.

Second, consider taking short breaks between work sessions to give your mind a chance to rest and recharge. You might use techniques like the Pomodoro Technique (25 minutes of focused work followed 
by a 5-minute break).

Third, prioritize your tasks based on urgency and importance. Focus on completing the most urgent or important task first before moving on to less critical ones.

Finally, try to be kind to yourself and recognize that it's normal to feel overwhelmed sometimes. It's okay if every single thing isn't completed by the end of the day as long as you're making 
progress towards your goals.

Would these steps help you get back to work?

>>> this might help
Great! Setting small, manageable goals and taking breaks can definitely make a difference in how productive you feel. Here's a quick summary of what you mentioned:

1. **Set Small Goals**: Break down tasks into smaller, more achievable steps.
2. **Take Short Breaks**: Use techniques like the Pomodoro Technique to rest your mind between work sessions.
3. **Prioritize Tasks**: Focus on urgent and important tasks first.
4. **Be Kind to Yourself**: Acknowledge that feeling overwhelmed is normal and recognize progress.

Now, how does that make you feel about getting back to work? Does this help clarify things for you?

>>> yea, kind of
That's a good start! If you're ready, we can chat more about it or if there's anything else on your mind.

>>> right now, I only have the productivity problem
Got it. Let's focus on productivity then. Have you tried any specific techniques in the past to improve your productivity? What worked well for you, and what didn't work as expected? Understanding 
what has worked before can help tailor better strategies moving forward.

>>> keeping my mind focussed have helped me alot
That sounds like a great strategy! Keeping your mind focused is key to improving productivity. Let's try to expand on that:

1. **Focus Techniques**: What specific techniques do you use to keep yourself focused? Do you find that certain environments or conditions help more than others?

2. **Eliminate Distractions**: Are there any common distractions you can identify and maybe reduce? For example, turning off notifications or finding a quiet workspace.

3. **Mindfulness and Breaks**: Sometimes taking short mindfulness breaks can actually improve focus for the longer sessions. Have you tried that before?

4. **Tools and Apps**: Are there any productivity tools or apps you use to stay on track? Things like to-do lists, timers, or focus apps might help.

Would you like to dive deeper into any of these strategies, or is there another aspect of productivity you want to explore?

>>> these works better.
Great! It's good to know that focusing techniques have worked well for you in the past. Let's refine and expand on those:

1. **Create a Focused Workspace**: Make sure your workspace is conducive to concentration. This might mean having a clean desk, minimizing clutter, or finding a quiet place where you can work without 
interruptions.

2. **Use Time Blocking**: Allocate specific time slots for different tasks throughout your day. This can help you stay focused and avoid multitasking, which often decreases productivity.

3. **Take Regular Breaks**: Even short breaks can refresh your mind. Consider techniques like the Pomodoro Technique (work for 25 minutes, then take a 5-minute break).

4. **Set Clear Goals**: Before starting work each day or session, set clear goals for what you want to achieve. This helps keep you motivated and focused on the task at hand.

5. **Eliminate Distractions**: Identify common distractions and try to minimize them. This could mean turning off notifications, using website blockers, or finding a quiet place to work.

6. **Mindfulness Practices**: Incorporating short mindfulness practices can improve your ability to focus. Even taking a few deep breaths or practicing meditation for 5-10 minutes before starting 
work can make a difference.

Would any of these strategies help you further improve your productivity?

>>> yea, I will try those.
That sounds like a solid plan! Let's summarize what we've discussed so far:

### Productivity Strategies:
1. **Create a Focused Workspace**: Ensure your workspace is clean and free from distractions.
2. **Use Time Blocking**: Allocate specific time slots for different tasks to avoid multitasking.
3. **Take Regular Breaks**: Use techniques like the Pomodoro Technique (25 minutes work, 5 minutes break).
4. **Set Clear Goals**: Define what you want to achieve before starting each work session.
5. **Eliminate Distractions**: Turn off notifications and minimize interruptions.
6. **Mindfulness Practices**: Incorporate short mindfulness breaks or deep breathing exercises.

### Next Steps:
1. Try implementing these strategies one by one over the next few days.
2. Pay attention to what works best for you and adjust as needed.

Would you like any further assistance with these strategies, or is there anything else on your mind?

"""

# Format the conversation
formatted_conversation = format_conversation(raw_conversation)

# Print the formatted conversation (optional)
print("Formatted Conversation:")
print(formatted_conversation)
print("\n" + "-"*50 + "\n")

# Evaluate the conversation
print("Sending to API for evaluation...")
evaluation_result = evaluate_conversation(formatted_conversation)

# Print the results
if evaluation_result:
    print("\nEvaluation Results:")
    print(json.dumps(evaluation_result, indent=2))
    
    # Print a summary of the scores
    print("\nCBT Dimension Scores:")
    for dimension, score in evaluation_result["target_scores"].items():
        print(f"  {dimension}: {score}")

if __name__ == "__main__":
    # If you want to evaluate a different conversation, you can replace raw_conversation
    # with your own conversation text or load it from a file
    pass

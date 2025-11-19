import smtplib
import ssl
from email.mime.text import MIMEText
import speech_recognition as sr
import pyttsx3
import config
import sys
import os
import tkinter as tk
from tkinter import simpledialog, scrolledtext, messagebox, ttk
import threading

def speak(text):
    """Convert text to speech using pyttsx3"""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass

def listen():
    """Listen to audio from microphone and convert to text using Google Speech Recognition"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

def normalize_email_address(text):
    """Convert speech patterns to proper email format"""
    # Convert common speech patterns
    text = text.lower()
    text = text.replace(" at the rate ", "@").replace(" at rate ", "@").replace("at rate", "@").replace("@", "@")
    text = text.replace(" dot ", ".").replace(" dot", ".").replace("dot ", ".")
    text = text.replace(" at ", "@")
    text = text.replace("gmail com", "gmail.com")
    text = text.replace("gmail.com", "gmail.com")
    text = text.replace(" ", "")  # Remove spaces from email
    return text

# Global live display window
live_display_window = None
live_display_text = None
live_display_data = {'to_addr': '', 'subject': '', 'body': ''}
mute_callback = None  # Callback to mute/unmute the main assistant

def create_live_display_window():
    """Create a window to display live email draft"""
    global live_display_window, live_display_text
    
    if live_display_window and live_display_window.winfo_exists():
        return live_display_window
    
    live_display_window = tk.Tk()
    live_display_window.title("📧 Live Email Draft")
    live_display_window.geometry("700x500")
    live_display_window.attributes('-topmost', True)
    
    # Header
    header = tk.Label(live_display_window, text="Live Email Draft Display", font=("Arial", 12, "bold"), bg="#E3F2FD")
    header.pack(fill=tk.X, padx=10, pady=10)
    
    # Display area
    frame = tk.Frame(live_display_window)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    live_display_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=20, width=80, font=("Arial", 11))
    live_display_text.pack(fill=tk.BOTH, expand=True)
    live_display_text.config(state=tk.DISABLED)
    
    return live_display_window

def update_live_display(to_addr=None, subject=None, body=None):
    """Update the live display window with current email details"""
    global live_display_window, live_display_text, live_display_data
    
    # Update global data with provided values
    if to_addr is not None:
        live_display_data['to_addr'] = to_addr
    if subject is not None:
        live_display_data['subject'] = subject
    if body is not None:
        live_display_data['body'] = body
    
    if not live_display_window or not live_display_window.winfo_exists():
        create_live_display_window()
    
    content = f"""
TO: {live_display_data['to_addr'] if live_display_data['to_addr'] else "(pending...)"}

SUBJECT: {live_display_data['subject'] if live_display_data['subject'] else "(pending...)"}

BODY:
{"-" * 70}
{live_display_data['body'] if live_display_data['body'] else "(pending...)"}
{"-" * 70}
    """
    
    try:
        live_display_text.config(state=tk.NORMAL)
        live_display_text.delete("1.0", tk.END)
        live_display_text.insert("1.0", content)
        live_display_text.config(state=tk.DISABLED)
        live_display_window.update()
    except Exception as e:
        print(f"Error updating display: {e}")

def close_live_display():
    """Close the live display window"""
    global live_display_window
    if live_display_window and live_display_window.winfo_exists():
        live_display_window.destroy()
        live_display_window = None

def get_input(prompt):
    """Prompt user via speech and listen for response"""
    speak(prompt)
    print(f"\n📢 AUDIO MODE - Waiting for your response to: {prompt}")
    print("⚠️  IMPORTANT: You are in MAIL COMPOSITION mode - speak your response (NOT a command)")
    print("-" * 80)
    value = listen()
    print(f"✓ You said: {value}")
    print("-" * 80)
    
    # Normalize if it's an email address
    if "email" in prompt.lower() or "mail" in prompt.lower() or "recipient" in prompt.lower():
        value = normalize_email_address(value)
        print(f"✓ Normalized to: {value}")
    
    return value

def process_body():
    """Process email body through voice commands with real-time display"""
    speak("You are now in mail composition mode. Speak your complete email body. Say EVA SEND when done.")
    print("\n" + "=" * 80)
    print("📧 EMAIL BODY COMPOSITION MODE - AUDIO INPUT")
    print("=" * 80)
    print("⚠️  IMPORTANT: Everything you say will be part of your email BODY")
    print("Commands available:")
    print("  - Say: 'EVA ENTER' or 'EVA NEXT LINE' for a new line")
    print("  - Say: 'EVA BACKSPACE' to remove the last word")
    print("  - Say: 'EVA SEND' when you're done with your email")
    print("=" * 80 + "\n")
    
    body = ""  # Complete body including all lines
    current_line = ""  # Current line being typed
    
    while True:
        print("🎤 Listening for your email content...")
        line = listen()
        line = line.strip()
        print(f"✓ Heard: {line}\n")
        
        if "eva send" in line.lower():
            # Add current line to body before sending
            if current_line:
                body += current_line
            print("✅ Email body complete!")
            break
        elif "eva enter" in line.lower() or "eva next line" in line.lower():
            # Add current line to body and move to next line
            body += current_line + "\n"
            current_line = ""
            print("📝 New line added\n")
            # Update live display with complete body including newline
            update_live_display(body=body)
        elif "eva backspace" in line.lower():
            # Remove last word from current line
            words = current_line.split()
            if words:
                removed = words.pop()
                current_line = " ".join(words)
                print(f"🔙 Removed word: '{removed}'\n")
            # Update live display with complete body
            update_live_display(body=body + current_line)
        else:
            # Add new words to current line
            if current_line:
                current_line += " " + line
            else:
                current_line = line
            print(f"📝 Current body preview: {(body + current_line)[:100]}...\n" if len(body + current_line) > 100 else f"📝 Current body: {body + current_line}\n")
            # Update live display with complete body
            update_live_display(body=body + current_line)
    
    return body

def get_dialog_input(prompt_title, prompt_message, is_multiline=False):
    """Show a GUI dialog box to get user input"""
    try:
        root = tk.Tk()
        root.withdraw()
        root.update()
        
        if is_multiline:
            print(f"DEBUG: Opening multiline dialog: {prompt_title}")
            dialog = tk.Toplevel(root)
            dialog.title(prompt_title)
            dialog.geometry("700x500")
            dialog.focus_set()
            dialog.attributes('-topmost', True)
            
            label = tk.Label(dialog, text=prompt_message, wraplength=650, justify=tk.LEFT, font=("Arial", 10))
            label.pack(pady=10, padx=10)
            
            text_area = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, height=18, width=80, font=("Arial", 10))
            text_area.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
            text_area.focus()
            
            result_container = []
            
            def on_ok():
                print("DEBUG: OK button clicked")
                result_container.append(text_area.get("1.0", tk.END).strip())
                dialog.destroy()
            
            def on_cancel():
                print("DEBUG: Cancel button clicked")
                result_container.append("")
                dialog.destroy()
            
            button_frame = tk.Frame(dialog)
            button_frame.pack(pady=10)
            
            ok_btn = tk.Button(button_frame, text="OK (Submit)", command=on_ok, width=15, font=("Arial", 10), bg="#4CAF50", fg="white")
            ok_btn.pack(side=tk.LEFT, padx=5)
            
            cancel_btn = tk.Button(button_frame, text="Cancel", command=on_cancel, width=15, font=("Arial", 10))
            cancel_btn.pack(side=tk.LEFT, padx=5)
            
            dialog.protocol("WM_DELETE_WINDOW", on_cancel)
            root.wait_window(dialog)
            result = result_container[0] if result_container else ""
            print(f"DEBUG: Multiline result length: {len(result)}")
        else:
            print(f"DEBUG: Opening single-line dialog: {prompt_title}")
            result = simpledialog.askstring(prompt_title, prompt_message, parent=root)
            result = result if result else ""
            print(f"DEBUG: Single-line result: {result}")
        
        try:
            root.destroy()
        except:
            pass
        
        return result
    except Exception as e:
        print(f"Dialog error: {e}")
        import traceback
        traceback.print_exc()
        return ""

def choose_input_mode():
    """Show a dialog to choose between audio and text input modes"""
    try:
        root = tk.Tk()
        root.title("EVA Mail - Input Mode Selection")
        root.geometry("400x200")
        
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        label = tk.Label(root, text="Choose Input Mode for Email Composition", font=("Arial", 12, "bold"))
        label.pack(pady=20)
        
        result = {'mode': 'text'}
        
        def on_audio():
            result['mode'] = 'audio'
            root.destroy()
        
        def on_text():
            result['mode'] = 'text'
            root.destroy()
        
        button_frame = tk.Frame(root)
        button_frame.pack(pady=20)
        
        audio_btn = tk.Button(button_frame, text="🎤 Audio Input", command=on_audio, width=20, font=("Arial", 11), bg="#4CAF50", fg="white")
        audio_btn.pack(pady=10)
        
        text_btn = tk.Button(button_frame, text="⌨️  Text Input", command=on_text, width=20, font=("Arial", 11), bg="#2196F3", fg="white")
        text_btn.pack(pady=10)
        
        root.mainloop()
        return result['mode']
    except Exception as e:
        print(f"Mode selection error: {e}")
        return 'text'  # Default to text (safer for mail composition)

def review_and_edit_email(to_addr, subject, body):
    """Show a review dialog where user can see and edit all email details"""
    try:
        root = tk.Tk()
        root.title("Review and Edit Email")
        root.geometry("800x650")
        
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        title_label = tk.Label(main_frame, text="Review Your Email Before Sending", font=("Arial", 12, "bold"))
        title_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Recipient field
        recipient_label = tk.Label(main_frame, text="To:", font=("Arial", 10, "bold"))
        recipient_label.pack(anchor=tk.W, pady=(10, 0))
        
        recipient_entry = tk.Entry(main_frame, font=("Arial", 10), width=100)
        recipient_entry.pack(fill=tk.X, padx=(20, 0), pady=(0, 10))
        recipient_entry.insert(0, to_addr)
        
        # Subject field
        subject_label = tk.Label(main_frame, text="Subject:", font=("Arial", 10, "bold"))
        subject_label.pack(anchor=tk.W, pady=(10, 0))
        
        subject_entry = tk.Entry(main_frame, font=("Arial", 10), width=100)
        subject_entry.pack(fill=tk.X, padx=(20, 0), pady=(0, 10))
        subject_entry.insert(0, subject)
        
        # Body field
        body_label = tk.Label(main_frame, text="Body:", font=("Arial", 10, "bold"))
        body_label.pack(anchor=tk.W, pady=(10, 0))
        
        body_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=15, width=100, font=("Arial", 10))
        body_text.pack(fill=tk.BOTH, expand=True, padx=(20, 0), pady=(0, 10))
        body_text.insert("1.0", body)
        
        result_container = {'action': None, 'to': to_addr, 'subject': subject, 'body': body}
        
        def on_send():
            print("DEBUG: Send button clicked")
            result_container['action'] = 'send'
            result_container['to'] = recipient_entry.get().strip()
            result_container['subject'] = subject_entry.get().strip()
            result_container['body'] = body_text.get("1.0", tk.END).strip()
            root.destroy()
        
        def on_cancel():
            print("DEBUG: Cancel button clicked")
            result_container['action'] = 'cancel'
            root.destroy()
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10, fill=tk.X)
        
        send_btn = tk.Button(button_frame, text="✓ Send Email", command=on_send, width=20, font=("Arial", 11), bg="#4CAF50", fg="white")
        send_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(button_frame, text="✕ Cancel", command=on_cancel, width=20, font=("Arial", 11), bg="#f44336", fg="white")
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        root.protocol("WM_DELETE_WINDOW", on_cancel)
        root.mainloop()
        
        if result_container['action'] == 'send':
            return (result_container['to'], result_container['subject'], result_container['body'])
        else:
            return (None, None, None)
    
    except Exception as e:
        print(f"Review dialog error: {e}")
        import traceback
        traceback.print_exc()
        return (None, None, None)

def send_mail_smtp(to_addr, subject, body, smtp_host, smtp_port, smtp_user, smtp_pass):
    """Send email via SMTP"""
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_addr
        context = ssl.create_default_context()
        
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls(context=context)
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_addr, msg.as_string())
        return True
    except Exception as e:
        print("Error sending email:", e)
        return False

def start_mail_composition(log_callback=None, mute_callback_func=None):
    """Start mail composition workflow with mode selection and review
    
    Args:
        log_callback: Optional callback function to log messages to UI
        mute_callback_func: Optional callback to mute/unmute the main assistant
    """
    global mute_callback
    mute_callback = mute_callback_func
    try:
        # Mute the main assistant while composing
        if mute_callback:
            print("🔇 Muting main assistant for mail composition...")
            mute_callback(True)
        
        # Show a disclaimer about mail mode
        disclaimer_msg = (
            "🚨 IMPORTANT: You are entering MAIL COMPOSITION MODE\n\n"
            "In this mode:\n"
            "- All input (voice or text) will be for your EMAIL\n"
            "- Commands like 'open app' or 'send message' will NOT be recognized\n"
            "- TEXT MODE is recommended for best results\n"
            "- If using AUDIO: speak clearly and avoid commands\n\n"
            "Click OK to continue, Cancel to exit."
        )
        messagebox.showinfo("Mail Composition Mode", disclaimer_msg)
        
        # Choose input mode first
        input_mode = choose_input_mode()
        print(f"\n{'='*80}")
        print(f"Selected input mode: {input_mode.upper()}")
        print(f"{'='*80}\n")
        
        if input_mode == 'audio':
            # Audio input mode
            create_live_display_window()
            
            to_addr = get_input("Who do you want to send the mail to?")
            update_live_display(to_addr=to_addr)
            if not to_addr:
                msg = "No recipient provided. Mail composition cancelled."
                messagebox.showwarning("Cancelled", msg)
                print(msg)
                close_live_display()
                return False
            
            subject = get_input("What is the subject?")
            update_live_display(to_addr=to_addr, subject=subject)
            if not subject:
                msg = "No subject provided. Mail composition cancelled."
                messagebox.showwarning("Cancelled", msg)
                print(msg)
                close_live_display()
                return False
            
            body = process_body()
            update_live_display(to_addr=to_addr, subject=subject, body=body)
            if not body:
                msg = "Empty email body. Mail composition cancelled."
                messagebox.showwarning("Cancelled", msg)
                print(msg)
                close_live_display()
                return False
            
            close_live_display()
        else:
            # Text input mode with dialogs
            to_addr = get_dialog_input(
                "Email Recipient",
                "Enter the recipient's email address:"
            )
            if not to_addr:
                msg = "No recipient provided. Mail composition cancelled."
                messagebox.showwarning("Cancelled", msg)
                print(msg)
                return False
            
            print(f"Recipient: {to_addr}")
            
            subject = get_dialog_input(
                "Email Subject",
                "Enter the email subject:"
            )
            if not subject:
                msg = "No subject provided. Mail composition cancelled."
                messagebox.showwarning("Cancelled", msg)
                print(msg)
                return False
            
            print(f"Subject: {subject}")
            
            print("DEBUG: About to open body dialog...")
            body = get_dialog_input(
                "Email Body",
                "Enter the email body (you can write multiple lines):",
                is_multiline=True
            )
            print(f"DEBUG: Body dialog returned {len(body)} characters")
            if not body:
                msg = "Empty email body. Mail composition cancelled."
                messagebox.showwarning("Cancelled", msg)
                print(msg)
                return False
            
            print(f"Body preview: {body[:100]}...")
        
        # Review and edit email before sending
        print("DEBUG: Opening review dialog...")
        review_result = review_and_edit_email(to_addr, subject, body)
        
        if review_result[0] is None:
            msg = "Email sending cancelled."
            messagebox.showinfo("Cancelled", msg)
            print(msg)
            return False
        
        to_addr, subject, body = review_result
        print(f"Final recipient: {to_addr}")
        print(f"Final subject: {subject}")
        print(f"Final body length: {len(body)} characters")
        
        # Get SMTP credentials from config
        smtp_host = getattr(config, 'SMTP_HOST', None) or 'smtp.gmail.com'
        smtp_port = getattr(config, 'SMTP_PORT', None) or 587
        smtp_user = getattr(config, 'SMTP_USER', None)
        smtp_pass = getattr(config, 'SMTP_PASSWORD', None)
        
        print(f"Sending email from {smtp_user} to {to_addr}...")
        
        # Send email
        result = send_mail_smtp(to_addr, subject, body, smtp_host, smtp_port, smtp_user, smtp_pass)
        
        if result:
            msg = "Mail sent successfully!"
            messagebox.showinfo("Success", msg)
            if log_callback:
                log_callback(msg)
            else:
                try:
                    speak(msg)
                except Exception:
                    pass
            print(msg)
            # Unmute assistant after successful send
            if mute_callback:
                print("🔊 Unmuting assistant after successful mail send")
                mute_callback(False)
        else:
            msg = "Failed to send mail. Check your credentials or internet connection."
            messagebox.showerror("Error", msg)
            if log_callback:
                log_callback(msg)
            else:
                try:
                    speak(msg)
                except Exception:
                    pass
            print(msg)
            # Unmute assistant after failed send
            if mute_callback:
                print("🔊 Unmuting assistant after mail send failed")
                mute_callback(False)
        
        return result
    
    except Exception as e:
        error_msg = f"Error in mail composition: {str(e)}"
        print(error_msg)
        messagebox.showerror("Error", error_msg)
        if log_callback:
            log_callback(error_msg)
        # Unmute assistant on error
        if mute_callback:
            print("🔊 Unmuting assistant due to error")
            mute_callback(False)
        return False

if __name__ == "__main__":
    start_mail_composition()

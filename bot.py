import os

import discord
from discord.ext import commands
from discord.ui import Button, View

intents = discord.Intents.default()
intents.guilds = True
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Mapping of button labels to role names and emoji colors
ROLE_MAP = [
    ("ONERA / Host", "🔴", discord.ButtonStyle.danger),
    ("Faculty / Professor", "🟡", discord.ButtonStyle.secondary),
    ("Postdoc / Researcher", "🟣", discord.ButtonStyle.secondary),
    ("PhD Candidate", "🔵", discord.ButtonStyle.primary),
    ("Master's Student", "🟢", discord.ButtonStyle.success),
    ("Bachelor's Student", "🟢", discord.ButtonStyle.success),
]

class RoleView(View):
    def __init__(self):
        super().__init__(timeout=None) # Persistent buttons
        for role_name, emoji, style in ROLE_MAP:
            # Custom button for each role
            button = Button(label=role_name, emoji=emoji, style=style, custom_id=f"role_{role_name}")
            button.callback = self.make_callback(role_name)
            self.add_item(button)

    def make_callback(self, role_name):
        async def button_callback(interaction: discord.Interaction):
            guild = interaction.guild
            role = discord.utils.get(guild.roles, name=role_name)
            
            if not role:
                await interaction.response.send_message(f"Role `{role_name}` not found!", ephemeral=True)
                return

            member = interaction.user

            # Toggle role: add if user doesn't have it, remove if they do
            if role in member.roles:
                await member.remove_roles(role)
                await interaction.response.send_message(f"Removed **{role.name}** role.", ephemeral=True)
            else:
                await member.add_roles(role)
                await interaction.response.send_message(f"Assigned **{role.name}** role! 🎉", ephemeral=True)

        return button_callback

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name}")

@bot.command()
@commands.has_permissions(administrator=True)
async def send_role_panel(ctx):
    # Delete the command trigger message to keep channel clean
    await ctx.message.delete()

    embed = discord.Embed(
        title="🎓 Academic & Professional Role Selection",
        description=(
            "Welcome to the **ONERA 468 CRM Wall Distribution Regression Challenge**!\n\n"
            "Please click a button below to select your academic or professional affiliation. "
            "Your username color and member category will update automatically.\n\n"
            "*(Clicking the button again will remove the role)*"
        ),
        color=discord.Color.blue()
    )

    view = RoleView()
    await ctx.send(embed=embed, view=view)

# Set via environment variable, never hardcode a bot token in source.
DISCORD_BOT_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
bot.run(DISCORD_BOT_TOKEN)

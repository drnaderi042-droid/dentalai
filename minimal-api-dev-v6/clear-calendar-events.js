const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient();

async function clearCalendarEvents() {
  try {
    console.log('Clearing all calendar events from database...');

    const result = await prisma.calendarEvent.deleteMany({});

    console.log(`✅ Successfully cleared ${result.count} calendar events`);

  } catch (error) {
    console.error('❌ Error clearing calendar events:', error);
  } finally {
    await prisma.$disconnect();
  }
}

clearCalendarEvents();

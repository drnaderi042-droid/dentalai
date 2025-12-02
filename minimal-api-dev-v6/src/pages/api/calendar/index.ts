import type { NextApiRequest, NextApiResponse } from 'next';

import { verify } from 'jsonwebtoken';
import { PrismaClient } from '@prisma/client';

import cors from 'src/utils/cors';

const prisma = new PrismaClient();
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

// ----------------------------------------------------------------------

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    await cors(req, res);

    // Verify authentication
    const { authorization } = req.headers;
    if (!authorization) {
      return res.status(401).json({ message: 'Authorization token missing' });
    }

    const accessToken = `${authorization}`.split(' ')[1];
    const data = verify(accessToken, JWT_SECRET) as { userId: string };

    if (!data.userId) {
      return res.status(401).json({ message: 'Invalid authorization token' });
    }

    // Get user info
    const user = await prisma.user.findUnique({
      where: { id: data.userId },
    });

    if (!user) {
      return res.status(401).json({ message: 'User not found' });
    }

    if (req.method === 'GET') {
      return handleGetEvents(req, res, data.userId);
    } if (req.method === 'POST') {
      return handleCreateEvent(req, res, data.userId);
    } if (req.method === 'PUT') {
      return handleUpdateEvent(req, res, data.userId);
    } if (req.method === 'PATCH') {
      return handleDeleteEvent(req, res, data.userId);
    } 
      return res.status(405).json({ message: 'Method not allowed' });
    
  } catch (error) {
    console.error('[Calendar API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  } finally {
    await prisma.$disconnect();
  }
}

async function handleGetEvents(req: NextApiRequest, res: NextApiResponse, userId: string) {
  // Get all events for the current user
  const events = await prisma.calendarEvent.findMany({
    where: {
      userId,
    },
    orderBy: {
      start: 'asc',
    },
  });

  // Format events for FullCalendar
  const formattedEvents = events.map(event => ({
    id: event.id,
    title: event.title,
    start: event.start.toISOString(),
    end: event.end.toISOString(),
    allDay: event.allDay,
    extendedProps: {
      description: event.description,
    },
    color: event.color,
    textColor: '#ffffff',
  }));

  return res.status(200).json({
    events: formattedEvents,
  });
}

async function handleCreateEvent(req: NextApiRequest, res: NextApiResponse, userId: string) {
  const { title, start, end, allDay = false, color = '#00a76f', description } = req.body;

  // Validate required fields
  if (!title || !start || !end) {
    return res.status(400).json({ message: 'Title, start, and end are required' });
  }

  // Create the event
  const event = await prisma.calendarEvent.create({
    data: {
      title,
      description,
      start: new Date(start),
      end: new Date(end),
      allDay,
      color,
      userId,
    },
  });

  // Return formatted event for FullCalendar
  const formattedEvent = {
    id: event.id,
    title: event.title,
    start: event.start.toISOString(),
    end: event.end.toISOString(),
    allDay: event.allDay,
    extendedProps: {
      description: event.description,
    },
    color: event.color,
    textColor: '#ffffff',
  };

  return res.status(201).json(formattedEvent);
}

async function handleUpdateEvent(req: NextApiRequest, res: NextApiResponse, userId: string) {
  const { id, title, start, end, allDay, color, description } = req.body;

  if (!id) {
    return res.status(400).json({ message: 'Event ID is required' });
  }

  // Verify the event belongs to the current user
  const existingEvent = await prisma.calendarEvent.findFirst({
    where: {
      id,
      userId,
    },
  });

  if (!existingEvent) {
    return res.status(404).json({ message: 'Event not found or access denied' });
  }

  // Update the event
  const updatedEvent = await prisma.calendarEvent.update({
    where: { id },
    data: {
      ...(title && { title }),
      ...(start && { start: new Date(start) }),
      ...(end && { end: new Date(end) }),
      ...(allDay !== undefined && { allDay }),
      ...(color && { color }),
      ...(description !== undefined && { description }),
    },
  });

  // Return formatted event for FullCalendar
  const formattedEvent = {
    id: updatedEvent.id,
    title: updatedEvent.title,
    start: updatedEvent.start.toISOString(),
    end: updatedEvent.end.toISOString(),
    allDay: updatedEvent.allDay,
    extendedProps: {
      description: updatedEvent.description,
    },
    color: updatedEvent.color,
    textColor: '#ffffff',
  };

  return res.status(200).json(formattedEvent);
}

async function handleDeleteEvent(req: NextApiRequest, res: NextApiResponse, userId: string) {
  const { eventId } = req.body;

  if (!eventId) {
    return res.status(400).json({ message: 'Event ID is required' });
  }

  // Verify the event belongs to the current user
  const existingEvent = await prisma.calendarEvent.findFirst({
    where: {
      id: eventId,
      userId,
    },
  });

  if (!existingEvent) {
    return res.status(404).json({ message: 'Event not found or access denied' });
  }

  // Delete the event
  await prisma.calendarEvent.delete({
    where: { id: eventId },
  });

  return res.status(200).json({ message: 'Event deleted successfully' });
}

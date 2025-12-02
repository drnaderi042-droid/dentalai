import { useMemo } from 'react';
import useSWR, { mutate } from 'swr';

import axios, { fetcher, endpoints } from 'src/utils/axios';

// ----------------------------------------------------------------------

const enableServer = true;

const CALENDAR_ENDPOINT = endpoints.calendar;

const swrOptions = {
  revalidateIfStale: enableServer,
  revalidateOnFocus: enableServer,
  revalidateOnReconnect: enableServer,
};

// ----------------------------------------------------------------------

export function useGetEvents() {
  const { data, isLoading, error, isValidating } = useSWR(CALENDAR_ENDPOINT, fetcher, swrOptions);

  const memoizedValue = useMemo(() => {
    const events = data?.events.map((event) => ({
      ...event,
      textColor: event.color,
    }));

    return {
      events: events || [],
      eventsLoading: isLoading,
      eventsError: error,
      eventsValidating: isValidating,
      eventsEmpty: !isLoading && !data?.events.length,
    };
  }, [data?.events, error, isLoading, isValidating]);

  return memoizedValue;
}

// ----------------------------------------------------------------------

export async function createEvent(eventData) {
  /**
   * Work on server
   */
  if (enableServer) {
    // Backend expects title, start, end directly in the body, not nested in eventData
    const data = {
      title: eventData.title,
      start: eventData.start,
      end: eventData.end,
      allDay: eventData.allDay || false,
      description: eventData.description || '',
      color: eventData.color || '#00a76f',
      ...(eventData.id && { id: eventData.id }),
    };
    
    // Validate required fields
    if (!data.title || !data.start || !data.end) {
      throw new Error('Title, start, and end are required');
    }
    
    await axios.post(CALENDAR_ENDPOINT, data);
  }

  /**
   * Work in local
   */
  mutate(
    CALENDAR_ENDPOINT,
    (currentData) => {
      const currentEvents = currentData?.events;

      const events = [...currentEvents, eventData];

      return { ...currentData, events };
    },
    false
  );
}

// ----------------------------------------------------------------------

export async function updateEvent(eventData) {
  /**
   * Work on server
   */
  if (enableServer) {
    // Validate that id exists
    if (!eventData?.id) {
      console.error('updateEvent called without id:', eventData);
      throw new Error('Event ID is required');
    }
    
    // Ensure id is a string (backend might expect string)
    const eventId = String(eventData.id);
    
    // Backend expects title, start, end directly in the body, not nested in eventData
    // For drag & drop operations, only id, start, end, and allDay are provided
    // So we only send the fields that are provided
    const data = {
      id: eventId,
    };
    
    // Only include fields that are provided
    if (eventData.title !== undefined) data.title = eventData.title;
    if (eventData.start !== undefined) data.start = eventData.start;
    if (eventData.end !== undefined) data.end = eventData.end;
    if (eventData.allDay !== undefined) data.allDay = eventData.allDay;
    if (eventData.description !== undefined) data.description = eventData.description;
    if (eventData.color !== undefined) data.color = eventData.color;
    
    // Validate required fields (only if they are being updated)
    if (data.title !== undefined && !data.title) {
      throw new Error('Title cannot be empty');
    }
    if (data.start !== undefined && !data.start) {
      throw new Error('Start date is required');
    }
    if (data.end !== undefined && !data.end) {
      throw new Error('End date is required');
    }
    
    console.log('Updating event with data:', data);
    await axios.put(CALENDAR_ENDPOINT, data);
  }

  /**
   * Work in local
   */
  mutate(
    CALENDAR_ENDPOINT,
    (currentData) => {
      const currentEvents = currentData?.events;

      const events = currentEvents.map((event) =>
        event.id === eventData.id ? { ...event, ...eventData } : event
      );

      return { ...currentData, events };
    },
    false
  );
}

// ----------------------------------------------------------------------

export async function deleteEvent(eventId) {
  /**
   * Work on server
   */
  if (enableServer) {
    const data = { eventId };
    await axios.patch(CALENDAR_ENDPOINT, data);
  }

  /**
   * Work in local
   */
  mutate(
    CALENDAR_ENDPOINT,
    (currentData) => {
      const currentEvents = currentData?.events;

      const events = currentEvents.filter((event) => event.id !== eventId);

      return { ...currentData, events };
    },
    false
  );
}

-- Run this in the Supabase SQL Editor
-- This will set up a pg_cron job to automatically increase the 'age' of crops by 1 every day at 8:00 AM.
-- Adjust the timezone or schedule if needed. The cron string '0 8 * * *' means 8:00 AM UTC. 
-- For 8:00 AM IST, you could use '30 2 * * *' (since IST is UTC+5:30)

-- First, ensure the pg_cron extension is enabled. (Can be done from the Supabase dashboard -> Database -> Extensions -> pg_cron)

SELECT cron.schedule(
    'update_crop_age_daily', -- Job name
    '30 2 * * *',            -- Schedule (At 08:00 AM IST which is 02:30 AM UTC)
    $$
        UPDATE crop_system
        SET age = age + 1
        WHERE current_stage != 'Harvested'; -- Optional: Don't increase age if harvested
    $$
);
